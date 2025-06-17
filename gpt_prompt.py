
INSTRUCTION = """Given the input consisting of a sentence, biomedical entity candidates with their classes, and related terms with semantic types retrieved from ontology by searching for the candidate text, select the most reasonable entity candidate if there is one. The selection criteria are that the candidate's word must be valid, and the assigned class must be appropriate for the candidate. Do not modify the candidate's word or class. If no suitable candidate is found, return no entity. Provide an explanation for the decision. Return the result in JSON format with the following fields: explanation, text, and class. The text and class fields in the output must exactly match the text and class of the selected candidate, and do not substitute them with the related terms and semantic types. The selected entity must be one of the provided candidates or none if no suitable candidate exists.

[Example 1]
INPUT:
-SENTENCE: \"Knockdown and overexpression of Psoriasin in pancreatic cancer cells was performed using specifically constructed plasmids , which either had anti - Psoriasin ribozyme transgene or the full length human Psoriasin coding sequence .\"
-CANDIDATES:
 1. \"full length human Psoriasin coding sequence\" (Gene or gene product)
 2. \"human\" (Organism taxon)
    -Homo sapiens (Human)

OUTPUT:
{
    \"explanation\": \"The most reasonable candidate is 'human' because it directly matches the related term 'Homo sapiens (Human).' In the sentence, 'human' specifies the source organism for the Psoriasin coding sequence. Although 'full length human Psoriasin coding sequence' is a valid candidate, it is more specific and does not align directly with the related term provided. Therefore, 'human' is the most appropriate choice based on the given related terms and the context of the sentence.\",
    \"text\": \"human\",
    \"class\": \"Organism taxon\"
}

[Example 2]
INPUT:
-SENTENCE: \"The results showed that OB counteracts most of the neurotransmitters changes caused by WRS .\"
-CANDIDATES:
 1. \"WRS\" (Chemical entity)
     -Romano-Ward Syndrome (Disease or Syndrome)
     -Wolcott-Rallison syndrome (Disease or Syndrome)
 2. \"WRS\" (Disease or phenotypic feature)
     -Romano-Ward Syndrome (Disease or Syndrome)
     -Wolcott-Rallison syndrome (Disease or Syndrome)

OUTPUT:
{
    \"explanation\": \"Neither candidate 'WRS' is suitable. The candidate 'WRS' labeled as a 'Chemical entity' is incorrect because 'WRS' refers to Wolcott-Rallison syndrome, which is a disease, not a chemical entity. The candidate 'WRS' labeled as a 'Disease or phenotypic feature' is closer to being correct but still does not fully align with the context of the sentence, which discusses neurotransmitter changes. Wolcott-Rallison syndrome, although a disease, does not have a direct or well-known connection to neurotransmitter changes. Therefore, no suitable candidate can be selected based on the provided information and context.\",
    \"text\": None,
    \"class\": None
}"""


def get_prompt(sentence, candidates, type_normalize={}):

    candidate_queries = []
    for i, c in enumerate(candidates):
        related_terms = []
        for t in c['related_terms']:
            related_terms.append(f"{t['name']} ({t['semantic_type']})")
        related_terms = f"{', '.join(related_terms)}"
        if len(related_terms) > 0:
            candidate_queries.append(f"Candidate {i+1}\n" \
                                 f"- Text: {c['text']}\n" \
                                 f"- Class: {type_normalize.get(c['type'], c['type'])}\n" \
                                 f"- Ontology matches: {related_terms}")
        else:
            candidate_queries.append(f"Candidate {i+1}\n" \
                                    f"- Text: {c['text']}\n" \
                                    f"- Class: {type_normalize.get(c['type'], c['type'])}")

    candidate_query = "\n\n".join(candidate_queries)
    query = f"SENTENCE: \"\"\"{sentence}\"\"\"\n{candidate_query}"

    return INSTRUCTION, query, candidate_query