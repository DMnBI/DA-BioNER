import json

import requests
import spacy
from scispacy.umls_linking import UmlsEntityLinker


def request_get(url, query=None):
    while True:
        try:
            output = requests.get(url, params=query)
            output.encoding = "utf-8"
            outputJson = json.loads(output.text)
            break
        except:
            print(url)
            print(query)
            print("Exception! Retry to request!")
            continue
    return outputJson


class UMLS_API:
    def __init__(self,
                 api_key=None,
                 version=None,
                 cache_file=None,
                 el_resolve_abbr=True,):
        self.api_key = api_key
        self.version = version
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.search_url = f"{self.base_url}/search/{self.version}"
        self.cui_url = f"{self.base_url}/content/{self.version}/CUI"
        self.cache_dict = None
        self.spacy_web_sm = None
        self.scispacy_md = None
        self.el_resolve_abbr = el_resolve_abbr

        if cache_file is not None:
            try:
                with open(cache_file, 'r', encoding="utf-8") as jfile:
                    self.cache_dict = json.load(jfile)
            except FileNotFoundError:
                print(f"Error: The file '{cache_file}' does not exist.")
            except json.JSONDecodeError:
                print(f"Error: The file '{cache_file}' is not a valid JSON file.")
        if self.cache_dict is None:
            self.cache_dict = {}

    def retrieval(self, text, return_def=True, n=1):
        all_results = {}
        all_results["linker"] = self.retrieval_linker(text, return_def, n)
        all_results["chunk_linker"] = self.retrieval_linker_w_chunk(text, return_def, n)

        final_result = []

        for key, values in all_results.items():
            for v in values:
                dup = False
                for i, item in enumerate(final_result):
                    if v["cui"] == item["cui"]:
                        final_result[i]["ref"].append(key)
                        dup = True
                        break
                if not dup and len(v) > 0:
                    v["ref"] = [key]
                    final_result.append(v)
        return final_result

    def retrieval_linker_w_chunk(self, text, return_def=True, n=1):
        umls_info = []
        if self.spacy_web_sm is None:
            self._load_spacy_web_sm()
        if self.scispacy_md is None:
            self._load_scispacy_md()
        doc = self.spacy_web_sm(text)
        for umls_ent in doc.noun_chunks:
            if len(umls_info) >= n: break
            chunk_text = umls_ent.text
            umls_info.extend(self.retrieval_linker(chunk_text, return_def, n))
        return umls_info[:min(n, len(umls_info))]

    def retrieval_linker(self, text, return_def=True, n=1):
        umls_info = []
        if self.scispacy_md is None:
            self._load_scispacy_md()
        linker = self.scispacy_md.get_pipe("scispacy_linker")
        scispacy_doc = self.scispacy_md(text)
        if scispacy_doc.ents:
            for each_extracted_chunk in scispacy_doc.ents:
                for i in range(len(each_extracted_chunk._.kb_ents)):
                    if len(umls_info) >= n: break
                    cui = each_extracted_chunk._.kb_ents[i][0]
                    if cui not in self.cache_dict:
                        self.cache_dict[cui] = {"cui": cui,
                                                "name": linker.kb.cui_to_entity[cui].canonical_name,
                                                "definition": linker.kb.cui_to_entity[cui].definition,
                                                "semantic_type": self._get_cui_information(cui)[0]}
                    if return_def:
                        umls_info.append(self.cache_dict[cui])
                    else:
                        umls_info.append(self.cache_dict[cui][:2]+[""]+self.cache_dict[-1:])
        return umls_info[:min(n, len(umls_info))]

    def _get_cui_information(self, cui, return_def=False):
        url = f"{self.cui_url}/{cui}"
        query = {'apiKey': self.api_key, }

        stype, definition = "", ""

        result_json = request_get(url, query)

        count = 0
        while 'result' not in result_json:
            if count >= 9:
                break
            print(f"Exception! cui '{cui}' has not result in UMLS!")
            result_json = request_get(url, query)
            count += 1
        if 'result' not in result_json:
            return stype, definition

        if 'semanticTypes' in result_json['result']:
            stype = result_json['result']['semanticTypes'][0]['name']
        if return_def:
            if 'definitions' in result_json['result'] \
                    and result_json['result']['definitions'] != "NONE":
                def_json = request_get(result_json['result']['definitions'], query)
                for tmp in def_json['result']:
                    if tmp['rootSource'] in ["MSH", "NCI"]:
                        definition = tmp['value']
                        break
                    if definition == "":
                        definition = def_json['result'][0]['value']

        return stype, definition

    def save_entity_cache(self, output_path):
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(self.cache_dict, json_file, ensure_ascii=False, indent=4)

    def _load_spacy_web_sm(self):
        self.spacy_web_sm = spacy.load("en_core_web_sm")

    def _load_scispacy_md(self):
        self.scispacy_md = spacy.load("en_core_sci_md")
        self.scispacy_md.add_pipe("merge_noun_chunks")
        linker = UmlsEntityLinker(resolve_abbreviations=True, filter_for_definitions=False)
        #self.scispacy_md.add_pipe(linker)
        self.scispacy_md.add_pipe("scispacy_linker",
                                  config={"linker_name": "umls",
                                          "resolve_abbreviations": self.el_resolve_abbr})
