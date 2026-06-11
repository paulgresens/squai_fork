import os
import time
import json
import requests
import gc
import torch
import numpy as np
from dotenv import load_dotenv
import time

load_dotenv()
# --- CONFIGURATION ---
SCADS_API_KEY = os.getenv("SCADS_API_KEY")

# --- CONFIGURATION ---
SCADS_API_MODEL = "openai/gpt-oss-120b"
INPUT_FILE = "./generatedUnjudged5.jsonl"
OUTPUT_FILE = "./generatedJudged5.jsonl"

JUDGING_PROMPT_TEMPLATE ="""
You are evaluating a scientific question-answer (Q-A) example.
Your goal is to assess whether it is a valid 2-hop, evidence-grounded scientific multi-hop question.
A valid example must:
-require combining information from exactly TWO papers
-involve dependent reasoning (Step 2 must require Step 1)
-not be decomposable into independent sub-questions
-be fully answerable from the provided evidence
-be self-contained and unambiguous

IMPORTANT:
-For each criterion, assign one label:
-GOOD = fully satisfies the criterion
-BORDERLINE = partially satisfies the criterion
-BAD = does not satisfy the criterion
-Be strict: only assign GOOD if the criterion is clearly and fully satisfied.

INPUT:
Question: {question} 
Answer: {answer} 
bridgeEvidencePaperText: {paper1Text}
bridgeAnswerPaperText: {paper2Text}

EVALUATION CRITERIA
A. Reasoning Structure
Multi-hop Validity: Does answering the question require combining both papers?
-GOOD: Both papers are strictly required; neither paper alone provides enough information to answer the complete question.
-BORDERLINE: Both papers contribute, but one paper may be sufficient to answer the main question, or the second paper mainly adds background/context.
-BAD: Only one paper is sufficient; the question is effectively single-hop.
Important: If one paper only identifies, introduces, defines, or names an artifact that the other paper already uses, mentions, evaluates, optimizes, cites, or compares against, this is not enough for GOOD. The answer must require combining non-redundant evidence from both papers.

Dependency Strength: Does Step 2 depend on the result of Step 1?
-GOOD: Step 2 strictly requires applying a proposition derived from Step 1. Step 1 must provide more than a named entity; it must provide a mechanism, condition, definition, limitation, assumption, guarantee, interpretation, objective, result, or similar.
-BORDERLINE: Step 2 is related to Step 1, but the dependency is weak, mostly entity-linking, or one step can mostly be solved without the other.
-BAD: Steps are independent, disconnected, or Step 1 only identifies a name/entity used in Step 2.

Important: If Step 1 can be solved mainly by matching a distinctive phrase in the question to nearly identical wording in one paper, and Step 2 only tracks the same artifact in the other paper, do not assign GOOD.

Non-Decomposability: Is the question NOT decomposable into independent sub-questions?
-GOOD: The question cannot be split into independent single-hop questions; solving one part changes, constrains, or enables solving the other.
-BORDERLINE: The question is partially decomposable, but the sub-answers still need some linking or interpretation.
-BAD: The question clearly decomposes into independent sub-questions whose answers can simply be concatenated.

B. Evidence Grounding
Evidence Distribution: Are both papers required and non-redundant?
-GOOD: Each paper contributes distinct, necessary information. One paper provides an intermediate proposition, and the other provides evidence whose meaning depends on applying that proposition.
-BORDERLINE: Both papers contribute, but there is overlap, redundancy, or one paper is dominant while the other only adds support.
-BAD: One paper is sufficient; the other is redundant, only background, or only provides the name/source of an artifact.

Very Important: Artifact reuse alone is insufficient. If Paper A introduces an artifact and Paper B merely uses, applies, evaluates, compares against, cites, optimizes, or extends that artifact, assign BAD or BORDERLINE unless the question requires applying a non-trivial proposition about that artifact.
If the answer can be produced by first identifying an artifact from Paper A and then looking up how Paper B uses that artifact, the Evidence grounding judgement has to be bad.

Answerability: Is the answer fully supported by the provided evidence?
-GOOD: Fully supported by the provided evidence from the papers; no external knowledge, speculation, or unsupported inference is needed.
-BORDERLINE: Partially supported, but some part of the answer requires mild inference, is under-specified, or is not directly grounded.
-BAD: Not supported, contradicted by the evidence, or requires external knowledge/speculation.

C. Dataset Quality: 
Decontextualization: Is the question self-contained and unambiguous?
The question cannot contain explicit references to the papers or their content such as "in this paper", "the proposed method", "this approach", "the authors", or similar.
-GOOD: Fully self-contained; all entities/concepts needed to understand the question are clearly named or described; no explicit paper references.
-BORDERLINE: Mostly self-contained, but contains minor ambiguity, an underspecified phrase, or a concept that is described but not clearly identifiable.
-BAD: Not understandable without external context, relies on paper-specific references, or is ambiguous.

Important: Technical terminology from the papers is allowed when necessary. However, the question should not simply copy distinctive phrasing from a paper in order to make Step 1 a lexical lookup.

OUTPUT FORMAT
{{
    "multiHopValidity": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the multiHopValidity criteria and justify jour judgement>"
    }},
    "dependencyStrength": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the dependencyStrength criteria and justify jour judgement>"
    }},
    "nonDecomposability": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the nonDecomposability criteria and justify jour judgement>"
    }},
    "evidenceDistribution": {{
        "judgement":"<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the evidenceDistribution criteria and justify jour judgement>"

    }},
    "answerability": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the answerability criteria and justify jour judgement>"

    }},
    "decontextualization": {{
        "judgement": "<GOOD / BORDERLINE / BAD>",
        "explanation": "<thoroughly describe how good the question fulfills the decontextualization criteria and justify jour judgement>"

    }},

    "confidence": <0 to 1 rate how confident you are in the judgements>
}}
Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
"""


MULTIHOP_VALIDITY_TEMPLATE ="""
You are evaluating a scientific question-answer (Q-A) example. You are given a generated question, an answer and two paper texts. Your goal is to judge how well this generated question fullfills a criterion.

The criterion is:
Multi-hop Validity: Does answering the question require combining the information from both papers?

1. Decompose the question into the necessary claims, that need to be answered in order to answer the whole question.
2. For each required claim, identify which papers answers it:
   -already stated in the question,
   -answerable only by Paper 1,
   -answerable only by Paper 2,
   -answerable by both papers,
   -unsupported or inferred.
3. Test whether either paper alone is sufficient:
   Could Paper 1 alone answer the complete question?
   Could Paper 2 alone answer the complete question?
4. Test whether question leakage makes one paper unnecessary:
   Does the question already state the main fact, mechanism, definition, objective, result, or condition that one paper is supposed to contribute?
   Could the other paper plus the information already stated in the question answer the main question?

5. Check whether both papers provide non-redundant required information:
   If one paper only identifies, introduces, defines, or names an artifact that the other paper already properly introduces, uses, mentions, evaluates, optimizes, cites, or compares against, this is not enough for GOOD. However, a good judgement here can be possible, if the other paper adds a non-trivial proposition about that artifact, required for interpreting, analyzing, applying or justifying or similar.
6. Distinguish the main answer from optional elaboration.
   Ask whether both papers are required for the central answer, not just for a richer or more detailed answer. If one paper answers the main question and the other only adds nuance, background, or a fuller mechanism, assign BORDERLINE rather than GOOD.

Finally, based on the previous steps, give a final judgement wether the question truly is multihop- valid.
JUDGEMENT OPTIONS:
-GOOD: Both papers are strictly required; neither paper alone provides enough information to answer the complete question.
-BORDERLINE: Both papers contribute, but one paper may be sufficient to answer the main question, or the second paper mainly adds background/context.
-BAD: Only one paper is sufficient; the question is effectively single-hop.
Important: If one paper only identifies, introduces, defines, or names an artifact that the other paper already uses, mentions, evaluates, optimizes, cites, or compares against, this is not enough for GOOD. The answer must require combining non-redundant evidence from both papers.

INPUT:
Question: {question}
Answer: {answer}
bridgeEvidencePaperText: {paper1Text}
bridgeAnswerPaperText: {paper2Text}

OUTPUT_FORMAT
{{
"thinking": {{
"claims": [
{{
"claim": <The first claim you identifid>,
"sourceSupportedBy": <question | paper1 | paper2 | both | unsupported_or_inferred>
}},
...
],
"paper1AloneSufficient": {{
"judgement": <true | false>,
"explanation": <explain why or why not paper 1 alone is sufficient for answering the question>
}},
"paper2AloneSufficient": {{
"judgement": <true | false>,
"explanation": <explain why or why not paper 2 alone is sufficient for answering the question>
}},
"paper1PlusQuestionSufficient": {{
"judgement": <true | false>,
"explanation": <explain why or why not paper 1 together with the information in the question is sufficient for answering the question>
}},
"paper2PlusQuestionSufficient": {{
"judgement": <true | false>,
"explanation": <explain why or why not paper 2 together with the information in the question is sufficient for answering the question>
}},
"questionLeakage": {{
"present": <true | false>,
"explanation": <explain why or why no question leakage is present>
}},
"redundancyOrArtifactOnly": {{
"present": <true | false>,
"explanation": <explain why or why not >
}},
"mainVsOptionalElaboration": {{
"present": <true | false>,
"explanation": <explain why or why not >
}}
}},
"judgement": "<insert the final judgement, considering all previous evaluations here. Are both papers really required. Answer with: GOOD | BORDERLINE | BAD>",
"explanation": "<thoroughly describe how good the question fulfills the multihop validity criteria and justify jour judgement based on previous evaluations>"
}}

Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
"""
DEPENDENCY_STRENGTH_TEMPLATE ="""
You are evaluating a scientific question-answer (Q-A) example. You are given a generated question, an answer, reasoning as well as two paper texts. Your goal is to judge how well this generated question fullfills a criterion.

The criterion is:
Dependency Strength: Does Step 2 depend on the result of Step 1?

1. Identify the Step 1 proposition.
   Ask: What exact mechanism, condition, definition, limitation, assumption, objective, result, or interpretation must be derived from Paper 1?

2. Check whether Step 1 is non-trivial.
   Step 1 should be more than identifying a method, artifact, dataset, metric, or named entity.

3. Check whether Step 1 is already leaked by the question.
   If the question already states the Step 1 proposition, then Step 2 does not strongly depend on deriving it from Paper 1.

4. Identify the Step 2 claim.
   Ask: What exact result, application, observation, comparison, or finding comes from Paper 2?

5. Test strict dependency.
   Ask: Could Step 2 be answered or interpreted without applying the Step 1 proposition?
   If yes, dependency is weak.

6. Check for entity-linking.
   If Step 1 only identifies an artifact/method and Step 2 only tracks that artifact/method in another paper, assign BAD or BORDERLINE, not GOOD.

Finally, based on the previous steps, give a final judgement wether the question consists of 2 consecutive steps, where step 2 is only answerable by applying the results of step 1.

JUDGEMENT OPTIONS:
-GOOD: Step 2 strictly requires applying a proposition derived from Step 1. Step 1 must provide more than a named entity; it must provide a mechanism, condition, definition, limitation, assumption, guarantee, interpretation, objective, result, or similar.
-BORDERLINE: Step 2 is related to Step 1, but the dependency is weak, mostly entity-linking, or one step can mostly be solved without the other.
-BAD: Steps are independent, disconnected, or Step 1 only identifies a name/entity used in Step 2.

INPUT:
Question: {question}
Answer: {answer}
reasoning : {reasoning}
bridgeEvidencePaperText: {paper1Text}
bridgeAnswerPaperText: {paper2Text}

OUTPUT_FORMAT
{{
"thinking": {{
"step1Claim": <Describe the mechanism you extracted from step1>,
"step1NotTrivial": {{
"judgement": <true | false>,
"explanation": <Describe why or why not step 1 is trivial>
}},
"step1LeakedByQuestion": {{
"judgement": <true | false>,
"explanation": <Describe why or why not step 1 is leaked by the question>
}},
"step2Claim": <Describe the mechanism you extracted from step2>,
"strictDependency": {{
"judgement": <true | false>,
"explanation": <Describe why or why not step2 is stricly dependant on step1>
}},
"entityLinking": {{
"judgement": <true | false>,
"explanation": <Describe why or why not the question is just entity linking>
}},
}},
"judgement": "<insert the final judgement, considering all previous evaluations here. Are both papers really required. Answer with: GOOD | BORDERLINE | BAD>",
"explanation": "<thoroughly describe how good the question fulfills the multihop validity criteria and justify jour judgement based on previous evaluations>"
}}

Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
"""
NON_DECOMPOSABILITY_TEMPLATE ="""
You are evaluating a scientific question-answer (Q-A) example. You are given a generated question, an answer and two paper texts. Your goal is to judge how well this generated question fullfills a criterion.

The criterion is: Non-Decomposability: Is the question NOT decomposable into independent sub-questions?

1. Ask what the answer for the question must explain, compare, justify, or conclude.
2. Try to decompose the question into independent sub-questions.
   For example:
   What does Paper 1 say about X?
   What does Paper 2 say about Y?
   What result does Paper 2 report?
   What mechanism does Paper 1 describe?
3. Check whether each sub-question can be answered independently.
   Ask whether the Paper 1 part and the Paper 2 part can each be solved without using the other.
4. Check whether the final answer is just concatenation or comparison.
   If the answer can be produced by separately retrieving facts from each paper and then simply joining them with “therefore,” “in contrast,” or “because,” the question is decomposable.
5. Check whether solving one part changes, constrains, or enables solving the other.
   GOOD requires that one sub-answer materially affects how the other sub-answer is interpreted or answered.
6. Check for artificial integration.
   If the question appears integrated only because it mentions both papers, both methods, or both results, but the reasoning can still be split into independent lookups, assign BAD or BORDERLINE.
7. Check whether the explanatory bridge is necessary.
   If the question asks for “why” or “how,” determine whether the bridge requires genuine synthesis or whether it is just a plausible comparison between independently retrieved facts.
   Decide whether the question is truly non-decomposable.
   If the answer requires integrated reasoning where one part constrains or enables the other, assign GOOD.
   If the answer needs some interpretation but the sub-answers are mostly independent, assign BORDERLINE.
   If the answer is mainly independent lookups plus concatenation/comparison, assign BAD.

Finally, based on the previous steps, give a final judgement wether the question is decomposable into independant sub-questions.
JUDGEMENT OPTIONS
-GOOD: The question cannot be split into independent single-hop questions; solving one part changes, constrains, or enables solving the other.
-BORDERLINE: The question is partially decomposable, but the sub-answers still need some linking or interpretation.
-BAD: The question clearly decomposes into independent sub-questions whose answers can simply be concatenated.

INPUT:
Question: {question}
Answer: {answer}
bridgeEvidencePaperText: {paper1Text}
bridgeAnswerPaperText: {paper2Text}

OUTPUT_FORMAT
{{
"thinking": {{
"mainQuestion":<Describe what the question must answer, explain, discuss, judge or similar>,
"subquestions": [
{{
"question": <The first question that was the result of you question decomposition >
"independentlyAnswerableBy": <give a verdict, wether the question can be independently solved by using either of the 3 options PAPER1 | PAPER2 | BOTH_PAPERS_TOGETHER>
}},
...
],
"concatenationOrComparison": {{
            "judgement": <true | false>,
            "explanation": "<explain whether the final answer can be produced by concatenating or simply comparing separately retrieved facts>"
        }},
"onePartConstrainsOther": {{
            "judgement": <true | false>,
            "explanation": "<explain whether solving one part changes, constrains, or enables solving the other>"
        }},
"artificialIntegration": {{
            "judgement": <true | false>,
            "explanation": "<explain whether the question only appears integrated because it mentions both papers, methods, entities, or results>"
        }},
"necessaryExplanatoryBridge": {{
            "judgement": <true | false>,
            "explanation": "<explain whether the explanatory bridge requires genuine synthesis or is just a plausible comparison>"
        }}
}},
"judgement": "<insert the final judgement, considering all previous evaluations here. Are both papers really required. Answer with: GOOD | BORDERLINE | BAD>",
"explanation": "<thoroughly describe how good the question fulfills the multihop validity criteria and justify jour judgement based on previous evaluations>"
}}

Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
"""
EVIDENCE_DISTRIBUTION_TEMPLATE ="""
You are evaluating a scientific question-answer (Q-A) example. You are given a generated question, an answer and two paper texts. Your goal is to judge how well this generated question fullfills a criterion.

The criterion is: Evidence Distribution: Are both papers required and non-redundant?

Important:

- The answer is a candidate answer, not evidence.
- Use the answer only as a clue to the intended response.
- Judge what evidence is actually required to answer the question.
- Verify all claims directly against the two paper texts.
- Be strict: assign GOOD only if both papers provide distinct, necessary, non-redundant evidence.

Judging procedure:

1. Identify the evidence needed to answer the question.
   Break the answer into evidence pieces, not just topics or method names.

2. List exactly what Paper 1 contributes.
   Ask whether Paper 1 provides a necessary intermediate proposition, mechanism, condition, definition, limitation, assumption, objective, result, or interpretation.
   Ask which of the evidence from 1. it contributes.

3. List exactly what Paper 2 contributes.
   Ask whether Paper 2 provides necessary evidence whose meaning depends on applying the Paper 1 contribution.
   Ask which of the evidence from 1. it contributes.

4. Check whether either paper is dominant.
   If one paper contains the task, result, comparison, method description, and enough context to answer the main question, while the other only adds support or detail, Evidence Distribution is not GOOD.

5. Check for redundancy or overlap.
   Ask whether both papers provide the same or overlapping information regarding the evidence needed for answering the question, or whether one paper repeats/elaborates what the other paper or the question already states.

6. Check for thin or artifact-only evidence roles.
   Ask whether one paper’s contribution is only naming, defining, or describing an artifact/method while the other paper provides the substantive task-specific evidence. If so, the evidence distribution is weak because one paper has only a background/detail role.

Finally, based on the previous steps, give a final judgement wether the question targets information distributed above both papers.
Judgement options:
-GOOD: Each paper contributes distinct, necessary information. One paper provides an intermediate proposition, and the other provides evidence whose meaning depends on applying that proposition.
-BORDERLINE: Both papers contribute, but there is overlap, redundancy, or one paper is dominant while the other only adds support.
-BAD: One paper is sufficient; the other is redundant, only background, or only provides the name/source of an artifact.

Input:
Question: {question}
Answer: {answer}
paper1Text: {paper1Text}
paper2Text: {paper2Text}

OUTPUT_FORMAT
{{
"thinking": {{
"evidenceNeeded": [
<List evidence that is needed to answer the question>
],
"evidencePaper1": [
<List evidence that paper 1 contribues>

        ],
        "evidencePaper2": [
            <List evidence that paper 2 contribues>
        ],
        "paperDominance": {{
            "dominantPaper": <answer with either PAPER1 | PAPER2 | NONE depending on if a paper is dominant or not>,
            "explanation": <explain why or why not either paper is dominant>
        }},
        "informationRedundancy": {{
            "judgement": <true | false>,
            "explanation": <explain why or why not the information from both papers is redundant>
        }},
        "artifactOnlyEvidence": {{
            "judgement": <true | false>,
            "explanation": <explain why or why not one paper's contribution is artifact only evidence>
        }}
    }},
    "judgement": "<insert the final judgement, considering all previous evaluations here. Does the question target evidence distributed across both papers? Answer with: GOOD | BORDERLINE | BAD>",
    "explanation": "<thoroughly describe how good the question fulfills the evidence distribution criteria and justify jour judgement based on previous evaluations>"
}}
Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json

"""
ANSWERABILITY_TEMPLATE = """
You are evaluating a scientific question-answer (Q-A) example. You are given a generated question, a candidate answer, and two paper texts. Your goal is to judge how well the candidate answer fulfills this criterion:

Answerability: Is the answer fully supported by the provided evidence?

Important:

- The answer is a candidate answer, not evidence.
- Verify the answer directly against the two paper texts.
- Do not use external knowledge.
- Be strict: assign GOOD only if all central parts of the answer are fully supported by the provided papers.
- If the answer contains an explanation such as “because,” “therefore,” “why,” “explains,” “accounts for,” or “makes X better/worse,” verify that this explanatory link is supported, not just the individual facts.

Judging procedure:

1. Break the candidate answer into the minimum claims needed to answer the question.
   Do not over-penalize irrelevant extra wording unless it affects the answer.

2. For each answer claim, identify whether it is supported by:
   - paper1
   - paper2
   - both
   - question_only
   - unsupported_or_inferred

3. Check whether every answer claim is directly supported by the paper texts.
   A good question must have the claims supported by the the paper texts, not by the question or unsopported or inferered.

4. Check whether any explanation or causal bridge is only inferred, instead of backed by evidence.

5. Check whether any answer claim is contradicted by the papers.

Finally, based on the previous steps, give a final judgement wether the question is fully answerable using the papers.

JUDGEMENT OPTIONS:

- GOOD: Fully supported by the provided evidence from the papers; no external knowledge, speculation, or unsupported inference is needed.
- BORDERLINE: Partially supported, but some part of the answer requires mild inference, is under-specified, or is not directly grounded.
- BAD: Not supported, contradicted by the evidence, or requires external knowledge/speculation.

INPUT:
Question: {question}
Answer: {answer}
bridgeEvidencePaperText: {paper1Text}
bridgeAnswerPaperText: {paper2Text}

OUTPUT FORMAT:
{{
  "thinking": {{
    "answerClaims": [
      {{
        "claim": "<The first claim extracted from the answer.>",
        "supportedBy" <insert by which source this claim is supported, answer with either:  paper1 | paper2 | both | question_only | unsupported_or_inferred>,
        "evidence: <explain why this claim is supported by the source>,
      }},
...
],
"allClaimsSupported": {{
      "judgement": <true | false>,
      "explanation": "<explain whether all claims are directly supported by the papers>"
    }},
"unsupportedInferenceOrBridge": {{
      "present": <true | false>,
      "explanation": "<explain whether the answer requires inference, an unsupported explanatory bridge, speculation, or external knowledge>"
    }},
"contradiction": {{
      "present": <true | false>,
      "explanation": "<explain whether any answer claim is contradicted by the papers>"
    }}
}},
"judgement": "<GOOD | BORDERLINE | BAD>",
"explanation": "<justify the Answerability judgement using your previous analysis from above>"
}}

Do not deviate from this schema. Do not add any preceding information such as ```json. Only answer with valid JSON.
"""
DECONTEXTUALIZATION_TEMPLATE = """
You are evaluating a scientific question-answer (Q-A) example. You are given a generated question, a candidate answer, and two paper texts. Your goal is to judge how well the generated question fulfills this criterion:

Decontextualization: Is the question self-contained and unambiguous?

Important:

- Judge the question itself, not the answer.
- Do not penalize technical terminology if the relevant entities, methods, datasets, metrics, or concepts are clearly named or described.
- Do not require the question to be easy for a layperson; it only needs to be understandable as a standalone scientific question.
- Be strict: assign GOOD only if the question can be understood without seeing the papers.
- Do not use this criterion to judge multi-hop validity, evidence distribution, or answerability.

Judging procedure:

1. Check whether the question contains explicit paper references.
   Look for phrases such as:
   - “in this paper”
   - “the proposed method”
   - “this approach”
   - “the authors”
   - “the study”
   - “the first paper”
   - “the second paper”
   - “the method described above”
   - “their results”
     If such references are needed to understand the question, the question is not fully decontextualized.

2. Check whether all key entities and concepts are named or clearly described.
   Ask whether the question clearly identifies the relevant methods, models, datasets, metrics, tasks, phenomena, or domain setting.

3. Check for ambiguous pronouns or underspecified references.
   Look for unclear uses of “it,” “this,” “that,” “the model,” “the method,” “the metric,” “the approach,” or similar.

4. Check for copied distinctive phrasing that turns the question into a lexical lookup.
   Technical terminology is allowed, but the question should not merely copy a long distinctive phrase from a paper to make retrieval trivial.

Finally, based on the previous steps, give a final judgement whether the question is self-contained and unambiguous.

Judgement options:

- GOOD: The question is understandable on its own, names or describes all required entities/concepts, and contains no paper-dependent references.
- BORDERLINE: The question is mostly self-contained but has minor ambiguity, underspecification, or slightly paper-dependent phrasing.
- BAD: The question relies on paper context, explicit paper references, ambiguous references, or is not understandable on its own.

Input:
Question: {question}
Answer: {answer}
paper1Text: {paper1Text}
paper2Text: {paper2Text}

Output format:
{{
  "thinking": {{
    "explicitPaperReferences": {{
      "judgement": <true | false>,
      "examples": [
        "<quote any explicit paper-dependent reference, or leave empty>"
      ],
      "explanation": "<explain whether the question contains explicit references to the papers or authors>"
    }},
"keyEntitiesAndConceptsNamed": {{
      "judgement": <true | false>,
      "missingOrUnclearItems": [
        "<state any missing or unclear entity/concept, or leave empty>"
      ],
      "explanation": "<explain whether the question names or clearly describes the necessary entities and concepts>"
    }},
"ambiguousReferences": {{
      "judgement": <true | false>,
      "examples": [
        "<quote any ambiguous pronoun/reference, or leave empty>"
      ],
      "explanation": "<explain whether the question contains ambiguous references>"
    }},
"hiddenPaperContextRequired": {{
      "present": <true | false>,
      "explanation": "<explain whether understanding the question requires seeing the papers>"
    }}
}},
"judgement": "<GOOD | BORDERLINE | BAD>",
"explanation": "<justify the Decontextualization judgement using the diagnostics above>"
}}

Do not deviate from this schema. Do not add any preceding information such as ```json. Only answer with valid JSON.
"""

EXPERIMENTERER_PROMPT = """
You are testing whether a scientific multi-hop question can be fully answered using only the provided papers.
You are given:
-a question
a subset of papers (one or more papers may be missing)

You are NOT checking whether the topic is mentioned.
You are checking whether the COMPLETE reasoning chain needed to answer the question is explicitly supported by the provided papers.

IMPORTANT:
-Use only the information from the given papers
-Do NOT rely on external knowledge
-Be strict: if any critical information is missing, the answer is NOT recoverable
-Every step required to answer must be explicitly supported by the text
-If ANY reasoning step is missing, mark as NOT answerable
-Mentioning related concepts is NOT sufficient
-Partial reasoning is NOT sufficient

INPUT:
Question: {question}
Papers: {paperTexts}

OUTPUT FORMAT:
{{
    "explanation": "<strict reasoning: list and explain which exact steps are supported and which are missing>",
    "answerable": <true/false - answer this as a boolean value based on your explanation on wether the question is fully answerable or not. Answer with true if it is, else with false>,
    "confidence":  <0 to 1 rate how confident you are>
}}
Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
"""

EXPERIMENTERER_CONNECTION_PROMPT ="""
You are given 2 scientific paper texts as well as a logical connection that exists between that papers. Your task is to determine wether that connection is explicitely mentioned in one of the 2 papers, or if is drawn based on their content, but not explicitely mentioned. 

IMPORTANT:
-Use only the information from the given papers
-Do NOT rely on external knowledge
-Be strict: if any critical information is missing, the answer is NOT recoverable
-Partial or speculative answers count as NOT answerable

INPUT:
paper1: {paper1}
paper2: {paper2}
connection: {connection}

OUTPUT FORMAT
{{
    "explanation": "<describe your reasoning>",
    "isPresent": <true/ false depending on if the connection is explicitely mentioned in at least one of the papers, false if not>
}}
Do not deviate from this schema. Do not add any preciding information like ```json. Only Answer with the valid json
"""

def build_judging_prompt(question,answer,paper1Text,paper2Text,reasoningSteps):
    return JUDGING_PROMPT_TEMPLATE.format(question=question,answer=answer,paper1Text=paper1Text,paper2Text=paper2Text)



def buildMultihopPrompt(question,answer,paper1Text,paper2Text):
    return MULTIHOP_VALIDITY_TEMPLATE.format(question=question,answer=answer,paper1Text=paper1Text,paper2Text=paper2Text)

def buildDependencyStrengthPrompt(question,answer,paper1Text,paper2Text, reasoning):
    return DEPENDENCY_STRENGTH_TEMPLATE.format(question=question,answer=answer,paper1Text=paper1Text,paper2Text=paper2Text,reasoning=reasoning)

def buildNonDecomposabilityPrompt(question,answer,paper1Text,paper2Text):
    return NON_DECOMPOSABILITY_TEMPLATE.format(question=question,answer=answer,paper1Text=paper1Text,paper2Text=paper2Text)

def buildEvidenceDistributionPrompt(question,answer,paper1Text,paper2Text):
    return EVIDENCE_DISTRIBUTION_TEMPLATE.format(question=question,answer=answer,paper1Text=paper1Text,paper2Text=paper2Text)

def buildAnswerabilityPrompt(question,answer,paper1Text,paper2Text):
    return ANSWERABILITY_TEMPLATE.format(question=question,answer=answer,paper1Text=paper1Text,paper2Text=paper2Text)

def buildDecontextualizationPrompt(question,answer,paper1Text,paper2Text):
    return DECONTEXTUALIZATION_TEMPLATE.format(question=question,answer=answer,paper1Text=paper1Text,paper2Text=paper2Text)




def buildExperimentererPromps(question,paperTexts):
    return EXPERIMENTERER_PROMPT.format(question=question, paperTexts=paperTexts)
def buildExperimentererConnectionPrompt(paper1, paper2,connection):
    return EXPERIMENTERER_CONNECTION_PROMPT.format(paper1=paper1, paper2=paper2, connection=connection)

def askScadsApiLLM(prompt):
    url = "https://llm.scads.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SCADS_API_KEY}"
    }

    # 3. Create your prompt payload
    payload = {
        "model": SCADS_API_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0
    }

    # 4. Send the request and print the answer
    response = requests.post(url, headers=headers, json=payload)
    # Convert the response to JSON and extract the text
    data = response.json()
    time.sleep(2)
    return (data["choices"][0]["message"]["content"])


def clean_and_parse_json(text):
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        json_str = text[start_idx : end_idx + 1]
        
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError as e:
            return None
    else:
        return None

def addJudgementToQuestion(question):
    start_time = time.perf_counter()
    cleanedAndParsedJson = question

    bridgeEvidencePaperText = cleanedAndParsedJson["bridgeEvidencePaperText"]
    bridgeAnswerPaperText = cleanedAndParsedJson["bridgeAnswerPaperText"]

    # judgementPrompt = build_judging_prompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answerWithoutPaperReferences"], bridgeEvidencePaperText, bridgeAnswerPaperText, json.dumps(cleanedAndParsedJson["reasoning"]))
    
    multihopPrompt = buildMultihopPrompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answerWithoutPaperReferences"], bridgeEvidencePaperText, bridgeAnswerPaperText)
    dependencyStrengthPrompt = buildDependencyStrengthPrompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answerWithoutPaperReferences"], bridgeEvidencePaperText, bridgeAnswerPaperText, json.dumps(cleanedAndParsedJson["reasoning"]))
    nonDecomposabilityPrompt = buildNonDecomposabilityPrompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answerWithoutPaperReferences"], bridgeEvidencePaperText, bridgeAnswerPaperText)
    evidenceDistributionPrompt = buildEvidenceDistributionPrompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answerWithoutPaperReferences"], bridgeEvidencePaperText, bridgeAnswerPaperText)
    answerabilityPrompt = buildAnswerabilityPrompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answerWithoutPaperReferences"], bridgeEvidencePaperText, bridgeAnswerPaperText)
    decontextualizationPrompt = buildDecontextualizationPrompt(cleanedAndParsedJson["question"], cleanedAndParsedJson["answerWithoutPaperReferences"], bridgeEvidencePaperText, bridgeAnswerPaperText)
    cleanedAndParsedJson["judgementResult"] = {}

    print("asking scads api - multihopResult")
    multihopResult = askScadsApiLLM(multihopPrompt)
    multihopResultParsed = clean_and_parse_json(multihopResult)
    if not multihopResultParsed: 
        return None
    cleanedAndParsedJson["judgementResult"]["multiHopValidity"] = multihopResultParsed

    print("multihopResult: " + multihopResultParsed["judgement"])

    time.sleep(1)
    print("asking scads api - dependencyStrengtResult")
    dependencyStrengtResult = askScadsApiLLM(dependencyStrengthPrompt)
    dependencyStrengtResultParsed = clean_and_parse_json(dependencyStrengtResult) 
    if not dependencyStrengtResultParsed: 
        return None
    cleanedAndParsedJson["judgementResult"]["dependencyStrength"] = dependencyStrengtResultParsed
    print("dependencyStrengtResult: " + dependencyStrengtResultParsed["judgement"])

    time.sleep(1)
    print("asking scads api - nonDecomposabilityResult")
    nonDecomposabilityResult = askScadsApiLLM(nonDecomposabilityPrompt)
    nonDecomposabilityResultParsed = clean_and_parse_json(nonDecomposabilityResult) 
    if not nonDecomposabilityResultParsed: 
        return None
    cleanedAndParsedJson["judgementResult"]["nonDecomposability"] = nonDecomposabilityResultParsed
    print("nonDecomposabilityResult: " + nonDecomposabilityResultParsed["judgement"])

    time.sleep(1)
    print("asking scads api - evidenceDistributionResult")
    evidenceDistributionResult = askScadsApiLLM(evidenceDistributionPrompt)
    evidenceDistributionResultParsed = clean_and_parse_json(evidenceDistributionResult) 
    if not evidenceDistributionResultParsed: 
        return None
    cleanedAndParsedJson["judgementResult"]["evidenceDistribution"] = evidenceDistributionResultParsed
    print("evidenceDistributionResult: " + evidenceDistributionResultParsed["judgement"])

    time.sleep(1)
    print("asking scads api - answerabilityResult")
    answerabilityResult = askScadsApiLLM(answerabilityPrompt)
    answerabilityResultParsed = clean_and_parse_json(answerabilityResult) 
    if not answerabilityResultParsed: 
        return None
    cleanedAndParsedJson["judgementResult"]["answerability"] = answerabilityResultParsed
    print("answerabilityResult: " + answerabilityResultParsed["judgement"])

    time.sleep(1)
    print("asking scads api - decontextualizationResult")
    decontextualizationResult = askScadsApiLLM(decontextualizationPrompt)
    decontextualizationResultParsed = clean_and_parse_json(decontextualizationResult) 
    if not decontextualizationResultParsed: 
        return None
    cleanedAndParsedJson["judgementResult"]["decontextualization"] = decontextualizationResultParsed
    print("decontextualizationResult: " + decontextualizationResultParsed["judgement"])
    time.sleep(1)
    
    # judgementResult = judgingAgent.generate(judgementPrompt)
    # judgementResult = askScadsApiLLM(judgementPrompt)
    # judgementResultParsed = clean_and_parse_json(judgementResult) 
    # print("JUDGE\n")
    # print(json.dumps(judgementResultParsed))
    # cleanedAndParsedJson["judgementResult"] = judgementResultParsed

    experimenterPromptEvidence = buildExperimentererPromps("paper1: " + cleanedAndParsedJson["question"], bridgeEvidencePaperText)
    experimenterPromptEvidenceExperimentorResult = askScadsApiLLM(experimenterPromptEvidence)
    experimenterPromptEvidenceExperimentorResultParsed = clean_and_parse_json(experimenterPromptEvidenceExperimentorResult) 
    if not experimenterPromptEvidenceExperimentorResultParsed: 
        return None

    experimenterPromptAnswer = buildExperimentererPromps("paper1: " + cleanedAndParsedJson["question"], bridgeAnswerPaperText)
    experimenterPromptAnswerExperimentorResult = askScadsApiLLM(experimenterPromptAnswer)
    experimenterPromptAnswerExperimentorResultParsed = clean_and_parse_json(experimenterPromptAnswerExperimentorResult) 
    if not experimenterPromptAnswerExperimentorResultParsed: 
        return None

    bothPaperTexts = "EvidencePaperText:\n" + bridgeEvidencePaperText + "\n" + "BridgeAnswerText" + "\n" + bridgeAnswerPaperText
    experimenterPromptBoth = buildExperimentererPromps(cleanedAndParsedJson["question"], bothPaperTexts)
    experimenterPromptBothResult = askScadsApiLLM(experimenterPromptBoth)
    experimenterPromptBothResultParsed = clean_and_parse_json(experimenterPromptBothResult) 
    if not experimenterPromptBothResultParsed: 
        return None

    cleanedAndParsedJson["experimenterPromptEvidenceExperimentorResult"] = experimenterPromptEvidenceExperimentorResultParsed
    cleanedAndParsedJson["experimenterPromptAnswerExperimentorResult"] = experimenterPromptAnswerExperimentorResultParsed
    cleanedAndParsedJson["experimenterPromptBothResult"] = experimenterPromptBothResultParsed
    # cleanedAndParsedJson["experimentererConnectionResult"] = experimentererConnectionResultParsed
    end_time = time.perf_counter()
    print("time for all requests: " + str(end_time - start_time))
    return cleanedAndParsedJson

def main():
    questionsWithoutJudgement = 0
    alreadyJudgedQuestions = []
    with open(OUTPUT_FILE, "r") as in_file:
        for j, lineAlready in enumerate(in_file):
            lineAlready = lineAlready.strip()
            if not lineAlready:
                continue # Skip empty lines
            question = clean_and_parse_json(lineAlready)
            alreadyJudgedQuestions.append(question["anchorPaper"])
    judgedQuestion = len(alreadyJudgedQuestions)

    with open(INPUT_FILE, "r") as in_file:
        for i, line in enumerate(in_file):
           
            line = line.strip()
            if not line:
                continue # Skip empty lines
            question = clean_and_parse_json(line)

            questionAnchorPaper = question["anchorPaper"]
            
            if questionAnchorPaper in alreadyJudgedQuestions:            
                print("Already judged question with anchor paper: " + questionAnchorPaper)
                continue
            else: 
                questionsWithoutJudgement += 1
                finishedQuestionWithJudgement = addJudgementToQuestion(question)
                print("Questions for anchor paper not judged yet: " + questionAnchorPaper)
            if (finishedQuestionWithJudgement):                                                    
                with open(OUTPUT_FILE, "a") as file: # Using "a" to append each new question
                    # json.dump automatically formats your dictionary and writes it to the file
                    json.dump(finishedQuestionWithJudgement, file)
                    file.write("\n") # Add a newline so the next JSON object starts on a new line
    print("-----------------------------")
    print("already judged:  " + str(judgedQuestion))
    print("added judgement: " + str(questionsWithoutJudgement))
    print("total:           " + str(judgedQuestion + questionsWithoutJudgement ))
    print("-----------------------------")        

if __name__ == "__main__":
    main()
