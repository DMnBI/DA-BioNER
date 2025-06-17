import argparse
import json
import os
import string
from collections import Counter

from openai import OpenAI
from tqdm import tqdm

from gpt_prompt import get_prompt
from umls import UMLS_API


class GPT:
    def __init__(
        self,
        api_key=None,
    ):

        self.client = OpenAI(api_key=api_key)

    def run_json(
        self,
        msg_list,
        version="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=4096,
    ):
        msgs = []
        for role, content in msg_list:
            msgs.append({"role": role,
                        "content": content})
        completion = self.client.chat.completions.create(
            messages=msgs,
            response_format={"type": "json_object"},
            model=version,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        chat_response = completion.choices[0].message.content
        chat_result = json.loads(chat_response)

        return chat_result


def iob2json(
    iob_file,
    retrieval,
    label_column,
):
    sent_words, entity_list = [], []
    char_idx, prev_tag = 0, "O"
    e_words, e_sidx= [] , -1
    result = []

    def get_entity_from_dictionary(words, s_idx, e_idx, type):
        e_text = " ".join(words)
        related_terms = retrieval.retrieval(e_text) if retrieval is not None else []
        return {"text": e_text,
                "start_idx": s_idx,
                "end_idx": e_idx,
                "type": type,
                "related_terms": related_terms}

    rf = open(iob_file, "r")
    for eachline in tqdm(rf.readlines(), desc="Reading iob file"):
        line_split = eachline.split()
        # print(line_split)
        if len(line_split) == 0:
            if e_sidx != -1:
                entity_list.append(get_entity_from_dictionary(e_words, e_sidx, char_idx - 1, prev_tag))
            result.append({"sentence": " ".join(sent_words), "entities": entity_list})
            sent_words, entity_list = [], []
            char_idx, prev_tag = 0, "O"
            e_words, e_sidx = [], -1
        elif len(line_split) >= 2:
            word, tag = line_split[0], line_split[label_column]
            if tag == "O":
                if e_sidx != -1:
                    entity_list.append(get_entity_from_dictionary(e_words, e_sidx, char_idx - 1, prev_tag))
                e_words, e_sidx = [], -1
            else:
                if (tag.split("-")[0] == "B" or prev_tag != tag.split("-")[-1]) and e_sidx != -1:
                    entity_list.append(get_entity_from_dictionary(e_words, e_sidx, char_idx - 1, prev_tag))
                    e_words, e_sidx = [], -1
                e_words.append(word)
                e_sidx = char_idx if e_sidx == -1 else e_sidx
            sent_words.append(word)
            char_idx += len(word) + 1
            prev_tag = tag.split("-")[-1]
    if e_sidx != -1:
        entity_list.append(get_entity_from_dictionary(e_words, e_sidx, char_idx - 1, prev_tag))
        result.append({"sentence": " ".join(sent_words), "entities": entity_list})
    rf.close()

    return result


def json2iob(
    json_file,
    output_path,
):

    with open(json_file, "r") as rf:
        ner_json = json.load(rf)

    with open(output_path, "w") as wf:
        for sent_info in ner_json:
            words = sent_info["sentence"].split()
            sidx = 0
            if len(sent_info["entities"]) > 0:
                e_iter = iter(sent_info["entities"])
                entity = e_iter.__next__()
            else:
                entity = None
            for word in words:
                eidx = sidx + len(word)
                wf.write(word)
                if None is not entity and entity["end_idx"] < sidx:
                    try:
                        entity = e_iter.__next__()
                    except:
                        entity = None
                if None is entity or ('type' in entity and entity['type'] == 'null'):
                    wf.write("\tO\n")
                elif entity["start_idx"] == sidx:
                    wf.write(f"\tB-{entity['type']}\n")
                elif entity["start_idx"] < sidx < eidx <= entity["end_idx"]:
                    wf.write(f"\tI-{entity['type']}\n")
                else:
                    wf.write("\tO\n")
                sidx = eidx + 1
            wf.write("\n")


def is_overlap(e1, e2):
    return e1["start_idx"] <= e2["start_idx"] <= e1["end_idx"] or \
        e2["start_idx"] <= e1["start_idx"] <= e2["end_idx"]


def is_equal_with_type(e1, e2):
    return e1["start_idx"] == e2["start_idx"] and \
        e1["end_idx"] == e2["end_idx"] and \
        e1["type"] == e2["type"]


def bundle(entity_list):
    merged_set = []
    sorted_entity_list = sorted(entity_list, key=lambda x: (x["start_idx"], x["end_idx"]))
    for i, e in enumerate(sorted_entity_list):
        is_merged = False
        if all(char in string.punctuation for char in e["text"]):
            continue
        for m_set in merged_set:
            if is_overlap(m_set, e):
                # add set
                has_equal_entity = False
                for elem in m_set["entities"]:
                    if is_equal_with_type(elem, e):
                        elem["ref_dataset"] += f",{e['ref_dataset']}"
                        has_equal_entity = True
                if not has_equal_entity:
                    m_set["entities"].append(e)
                m_set["start_idx"] = min(m_set["start_idx"], e["start_idx"])
                m_set["end_idx"] = max(m_set["end_idx"], e["end_idx"])
                is_merged = True
                break
        if not is_merged:
            merged_set.append({
                "start_idx": e["start_idx"],
                "end_idx": e["end_idx"],
                "entities": [e],
            })
    return merged_set


def merge_overlap_entities(json_list):
    merged_json = []
    for i in range(len(json_list[0][1])):
        all_entities = []
        sentence = json_list[0][1][i]["sentence"]
        for ref_name, e_json in json_list:
            if e_json[i]["sentence"] != sentence:
                print(f"Error! Sentences are different (idx: {i})")
                print(sentence)
                print(e_json[i]["sentence"])
                exit()
            for e in e_json[i]["entities"]:
                e["ref_dataset"] = ref_name
            all_entities.extend(e_json[i]["entities"])
        merged_entity_list = bundle(all_entities)
        merged_json.append({"sentence": sentence,
                            "merged_entities": merged_entity_list,})
    return merged_json


def run_gpt(gpt_api, prompt, query, gpt_ver, temperature):
    run_limit = 3
    while True:
        try:
            result = gpt_api.run_json([["system", prompt], ["user", query]], version=gpt_ver, temperature=temperature)
        except json.decoder.JSONDecodeError:
            print("JsonDecodeError")
            if run_limit == 0:
                break
            run_limit -= 1
            continue
        try:
            final_text, final_type, explanation \
                = result["text"], result["class"], result["explanation"]
        except:
            print(f"Format Error: {result}")
            print(f"Prompt: {prompt}")
            print(f"Query: {query}")
            print(f"Try again!")
            if run_limit == 0:
                break
            run_limit -= 1
            continue
        break


    if explanation is None:
        final_text, final_type, explanation = "None", "None", f"Result Error: {result}"

    return final_text, final_type, explanation

def gpt_ensemble(
    candidate_json,
    output_path,
    api_key,
    version="gpt-3.5-turbo-0125",
    temperature=0.7,
    iteration=3,
    type_normalize={},
):
    gpt_api = GPT(api_key=api_key)

    ensemble_json = []
    num_groups, num_mult_cand_groups = 0, 0
    for sent in tqdm(candidate_json):
        new_sent = {}
        sentence = sent["sentence"]
        new_sent["sentence"] = sent["sentence"]
        final_entities = []
        for group in sent["merged_entities"]:
            num_groups += 1
            if len(group["entities"]) >= 2:
                num_mult_cand_groups += 1

            prompt, query, candidate_query = get_prompt(sentence, group["entities"],type_normalize)
            candidates, explanations = [], []

            for i in range(iteration):
                final_text, final_type, explanation = run_gpt(gpt_api, prompt, query, version, temperature)

                added = False
                if final_text is not None and final_type is not None:
                    def norm_text(text):
                        return text

                    for c in group["entities"]:
                        if norm_text(c["text"]) == norm_text(final_text): final_text = c['text']
                        if c["text"] == final_text and type_normalize.get(c["type"], c["type"]).lower() == final_type.lower():
                            candidates.append(c)
                            added = True
                            break
                if not added:
                    candidates.append({'text': "None", 'type': "None",
                                       "related_terms": [c["related_terms"] for c in group["entities"]]})
                explanations.append([explanation, final_text, final_type])
            counter = Counter((c['text'], c['type']) for c in candidates)
            majority_entity = sorted(counter.items(), key=lambda item: (-item[1], len(item[0][0])))
            final_entity = None
            for e in group["entities"]:
                try:
                    if e['text'] == majority_entity[0][0][0] and e["type"] == majority_entity[0][0][1]:
                        final_entity = e
                        break
                except IndexError:
                    final_entity = None
            if final_entity is None:
                final_entity = {}
                final_entity["text"] = None
                final_entity["start_idx"] = -group['start_idx']
                final_entity["end_idx"] = -group['end_idx']
            final_entity["candidates"] = candidate_query
            final_entity["gpt_results"] = majority_entity
            final_entity["gpt_explanations"] = explanations
            final_entities.append(final_entity)

        new_sent["entities"] = final_entities
        ensemble_json.append(new_sent)
        with open(output_path, "w", encoding="utf-8") as wf:
            json.dump(ensemble_json, wf, ensure_ascii=False, indent=4)

    return ensemble_json


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_file")
    parser.add_argument("--output_dir")
    parser.add_argument("--api_key")
    parser.add_argument("--version", default="gpt-3.5-turbo-0125")
    parser.add_argument("--temperature", default=0.7)
    parser.add_argument("--num_gpt_calls", default=3)
    parser.add_argument("--umls_version", default="2023AA")
    parser.add_argument("--umls_api_key", default=None)
    parser.add_argument("--umls_cache", default=None)
    parser.add_argument("--label_column", default=-1)
    parser.add_argument("--overwrite_preprocessed_file", action="store_true")
    parser.add_argument("--overwrite_gpt_output", action="store_true")
    parser.add_argument("--type_norm_file", default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    type_norm_dict = json.load(open(args.type_norm_file, "r")) if args.type_norm_file is not None else {}

    umls_api = UMLS_API(api_key=args.umls_api_key, version=args.umls_version,
                        cache_file=args.umls_cache) if args.umls_api_key is not None else None
    json_list = []

    candidate_path = os.path.join(args.output_dir, f"candidates.json")
    ensemble_json_path = os.path.join(args.output_dir, "ensemble.json")
    ensemble_iob_path = os.path.join(args.output_dir, "ensemble.iob.txt")

    if (os.path.exists(ensemble_json_path) or os.path.exists(ensemble_iob_path)) and not args.overwrite_gpt_output:
        print(
            f"The ensemble file ({ensemble_json_path}, {ensemble_iob_path}) already exists. To reprocess and overwrite it, "
            f"please use the --overwrite_gpt_output argument.")
        exit()

    if not os.path.exists(candidate_path) or args.overwrite_preprocessed_file:
        with open(args.list_file, "r") as rf:
            iob_list = rf.readlines()
        for i, iob_path in enumerate(iob_list):
            iob_path = iob_path.strip()
            json_path = os.path.join(args.output_dir, f"model{i+1}.json")
            if not os.path.exists(json_path) or args.overwrite_preprocessed_file:
                json_result = iob2json(iob_path, umls_api, args.label_column)
                with open(json_path, "w", encoding="utf-8") as jw:
                    json.dump(json_result, jw, ensure_ascii=False, indent=4)
            else:
                json_result = json.load(open(json_path, "r"))
            json_list.append([f"model{i+1}", json_result])

        candidate_json = merge_overlap_entities(json_list)
        with open(candidate_path, "w", encoding="utf-8") as mw:
            json.dump(candidate_json, mw, ensure_ascii=False, indent=4)
    else:
        candidate_json = json.load(open(candidate_path, "r"))

    ensemble_json = gpt_ensemble(candidate_json, ensemble_json_path, api_key=args.api_key,
                                 iteration=args.num_gpt_calls, version=args.version,
                                 temperature=args.temperature, type_normalize=type_norm_dict)
    json2iob(ensemble_json_path, ensemble_iob_path)
