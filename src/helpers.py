from dataclasses import dataclass, field
import typing as ty
import pandas as pd
import random
import re
from datetime import datetime
import os
from openai import OpenAI
from llama_cpp import Llama
import sys

PATTERNS_COLUMNS = [
    "Gender",
    "Is-Political",
    "Is-Formal",
    "Pattern-Language",
    "Pattern-ID",
    "Political-Leaning",
    "Is-Valid",
    "Conclusion",
    "Conclusion-Verblast",
    "Premises",
    "Premises-Verblast"
]

INSTRUCTIONS_COLUMNS = [
    "Instruction-Language",
    "Is-Formal",
    "Instruction-ID",
    "Instruction"
]

ARGUMENTS_COLUMNS = [
    "Gender",
    "Is-Political",
    "Is-Formal",
    "Pattern-Language",
    "Pattern-ID",
    "Political-Leaning",
    "Is-Valid",
    "Variation-ID",
    "Argument"
]

PROMPTS_COLUMNS = [
    "Pattern-Language",
    "Is-Political",
    "Is-Formal",
    "Pattern-ID",
    "Gender",
    "Political-Leaning",
    "Is-Valid",
    "Variation-ID",
    "Instruction-Language",
    "Instruction-ID",
    "Is-Few-Shots",
    "Few-Shots-Setting-Idx",
    "Prompt"
]

RESULTS_COLUMNS = [
    "Gender",
    "Is-Political",
    "Is-Formal",
    "Pattern-Language",
    "Pattern-ID",
    "Political-Leaning",
    "Is-Valid",
    "Variation-ID",
    "Instruction-Language",
    "Instruction-ID",
    "Is-Few-Shots",
    "Few-Shots-Setting-Idx",
    "Prompt",
    "Created-At",
    "Model-Name",
    "Output",
    "Bare-Grade"
]

LANGUAGE_VALUES = ["en", "dech"]
IS_FORMAL_VALUES = [1, 0]
POLITICAL_LEANING_VALUES = ["left", "right"]
POLITICAL_LEANING_VALUES_NON_POLITICAL = ["poker", "chess"]
VARIATION_ID_VALUES = ["default", "perm", "rand", "conlast"]
GENDER_VALUES = ["f", "m"]
IS_VALID_VALUES = [0, 1]

POLITICS_DICT = {
    "en": {
        "left": {
            "PARTY": "Democratic",
            "LEADER": "Joe Biden"
        },
        "right": {
            "PARTY": "Republican",
            "LEADER": "Donald J. Trump"
        }
    },
    "dech": {
        "left": {
            "PARTY": "SP",
            "LEADER": "Beat Jans"
        },
        "right": {
            "PARTY": "SVP",
            "LEADER": "Albert Rösti"
        }
    }
}

TEMPLATE_FEW_SHOTS_EN_1 = """<INSTRUCTION_0>. Here's a couple of examples to help you understand what you should do:
<EXAMPLE_ARGUMENTS>
Now, <INSTRUCTION_1>"""

TEMPLATE_FEW_SHOTS_DECH_1 = """<INSTRUCTION_0>. Hier sind einige Beispiele, die Dir dabei helfen, zu verstehen, was Du machen sollst:
<EXAMPLE_ARGUMENTS>
Jetzt bist Du dran: <INSTRUCTION_1>"""

TEMPLATES_FEW_SHOTS = {
    1: {
        "en": TEMPLATE_FEW_SHOTS_EN_1,
        "dech": TEMPLATE_FEW_SHOTS_DECH_1
    }
}

@dataclass
class FewShotSettings:
    language_values: ty.List[str] = field(default_factory=lambda: LANGUAGE_VALUES)
    num_language_values: int = len(LANGUAGE_VALUES)

    is_formal_values: ty.List[int] = field(default_factory=lambda: IS_FORMAL_VALUES)
    num_is_formal_values: int = len(IS_FORMAL_VALUES)

    political_leaning_values: ty.List[str] = field(default_factory=lambda: POLITICAL_LEANING_VALUES)
    num_political_leaning_values: int = len(POLITICAL_LEANING_VALUES)

    variation_id_values: ty.List[str] = field(default_factory=lambda: VARIATION_ID_VALUES)
    num_variation_id_values: int = len(VARIATION_ID_VALUES)

    gender_values: ty.List[str] = field(default_factory=lambda: GENDER_VALUES)
    num_gender_values: int = len(GENDER_VALUES)

    is_valid_values: ty.List[int] = field(default_factory=lambda: IS_VALID_VALUES)
    num_is_valid_values: int = len(IS_VALID_VALUES)

    sort_random: bool = False
    sort_values_parameters: ty.List[ty.Tuple[str, bool]] = field(default_factory=lambda: [
        ('Pattern-Language', False),
        ('Is-Political', False), 
        ('Is-Formal', False),
        ('Gender', True), 
        ('Political-Leaning', True), 
        ('Is-Valid', True), 
        ('Variation-ID', True),
    ])

    @staticmethod
    def _validate_values_and_num_values(values_list, num_values):
        if len(values_list) < num_values:
            raise ValueError(f"len(<values_list>) must be greater than or equal to <num_values>. Found:\n"
                             f"{len(values_list)=}\n"
                             f"{num_values=}")
        
    def _split_sort_values_parameters(self):
        self.sort_values_by = []
        self.sort_values_ascending = []

        for by, ascending in self.sort_values_ascending:
            self.sort_values_by.append(by)
            self.sort_values_ascending.append(ascending)

    def __post_init__(self):
        self._validate_values_and_num_values(self.language_values, self.num_language_values)
        self._validate_values_and_num_values(self.is_formal_values, self.num_is_formal_values)
        self._validate_values_and_num_values(self.political_leaning_values, self.num_political_leaning_values)
        self._validate_values_and_num_values(self.variation_id_values, self.num_variation_id_values)
        self._validate_values_and_num_values(self.gender_values, self.num_gender_values)
        self._validate_values_and_num_values(self.is_valid_values, self.num_is_valid_values)
        
        self._split_sort_values_parameters()

def truly_shuffle(list_to_shuffle):
    list_to_shuffle_copy = list_to_shuffle.copy()

    while list_to_shuffle == list_to_shuffle_copy:
        random.shuffle(list_to_shuffle_copy)

    return list_to_shuffle_copy

def build_argument(pattern_row, variation_id):
    argument = pattern_row["Conclusion"] # In German use Conclusion_verblast?
    
    if pattern_row["Pattern-Language"] == "en":
        random_phrase = ", because the sun rises every day"
        before_last_premise = ", and because "
        between_premises = ", because "
        before_last_premise_case_single_premise = " because "
        premises_list = pattern_row["Premises"].split(";")
    elif pattern_row["Pattern-Language"] == "dech":
        random_phrase = ", weil die Sonne jeden Tag scheint"
        before_last_premise = ", und weil "
        between_premises = ", weil "
        before_last_premise_case_single_premise = " weil "
        premises_list = pattern_row["Premises-Verblast"].split(";")
    else:
        raise ValueError(f"Unknown Pattern-Language. Found {pattern_row['Pattern-Language']=}")

    if variation_id == "perm":
        premises_list = truly_shuffle(premises_list)

    for idx, premise in enumerate(premises_list):
        if idx == len(premises_list)-1:
            if variation_id == "rand":
                argument = f"{argument}{random_phrase}"

            if len(premises_list) == 1 and variation_id != "rand":    
                argument = f"{argument}{before_last_premise_case_single_premise}{premise}"
            else:
                argument = f"{argument}{before_last_premise}{premise}"
        else:
            argument = f"{argument}{between_premises}{premise}"

    argument = f"{argument}."

    return argument

def build_argument_conlast(pattern_row):
    premises_list = pattern_row["Premises"].split(";")
    
    if pattern_row["Pattern-Language"] == "en":
        before_conclusion = ". Therefore, "
    elif pattern_row["Pattern-Language"] == "dech":
        before_conclusion = ". Aus dem ergibt sich: "
    else:
        raise ValueError(f"Unknown Pattern-Language. Found {pattern_row['Pattern-Language']=}")

    argument = ""

    for idx, premise in enumerate(premises_list):
        if idx == 0:
            argument = f"{argument}{premise[0].upper()}{premise[1:]}"
        else:
            argument = f"{argument}. {premise[0].upper()}{premise[1:]}"

    argument = f"{argument}{before_conclusion}{pattern_row['Conclusion']}."

    return argument


def generate_arguments_all(patterns):
    arguments_all = pd.DataFrame(columns=ARGUMENTS_COLUMNS)
    
    for _, pattern_row in patterns.iterrows():
        for variation_id in VARIATION_ID_VALUES:
            if variation_id == "conlast":
                argument = build_argument_conlast(pattern_row)
            else:
                argument = build_argument(pattern_row, variation_id)
            
            argument_series = pd.Series({
                "Gender": pattern_row["Gender"],
                "Is-Political": pattern_row["Is-Political"],
                "Is-Formal": pattern_row["Is-Formal"],
                "Pattern-Language": pattern_row["Pattern-Language"],
                "Pattern-ID": pattern_row["Pattern-ID"],
                "Political-Leaning": pattern_row["Political-Leaning"],
                "Is-Valid": pattern_row["Is-Valid"],
                "Variation-ID": variation_id,
                "Argument": argument
            })

            arguments_all.loc[len(arguments_all)] = argument_series

    return arguments_all.reset_index(drop=True)

def generate_arguments_few_shots(arguments_all, few_shot_setting, random_seed):    
    arguments_few_shots_samples_df_list = []

    language_values_effective = random.sample(few_shot_setting.language_values, few_shot_setting.num_language_values)
    for language in language_values_effective:
        is_formal_values_effective = random.sample(few_shot_setting.is_formal_values, few_shot_setting.num_is_formal_values)
        for is_formal in is_formal_values_effective:
            leaning_values_effective = random.sample(few_shot_setting.political_leaning_values, few_shot_setting.num_political_leaning_values)
            print(leaning_values_effective)
            for leaning in leaning_values_effective:
                gender_values_effective = random.sample(few_shot_setting.gender_values, few_shot_setting.num_gender_values)
                for gender in gender_values_effective:
                    is_valid_values_effective = random.sample(few_shot_setting.is_valid_values, few_shot_setting.num_is_valid_values)
                    for is_valid in is_valid_values_effective:
                        variation_id_values_effective = random.sample(few_shot_setting.variation_id_values, few_shot_setting.num_variation_id_values)
                        for variation_id in variation_id_values_effective:
                            cond = (arguments_all["Is-Formal"]==is_formal) & \
                                    (arguments_all["Gender"]==gender) & \
                                    (arguments_all["Pattern-Language"]==language) & \
                                    (arguments_all["Is-Valid"]==is_valid) & \
                                    (arguments_all["Political-Leaning"]==leaning) & \
                                    (arguments_all["Variation-ID"]==variation_id)
                            
                            # print(language, is_formal, leaning, gender, is_valid, variation_id)
                            
                            df = arguments_all[cond].sample(1, random_state=random_seed)
                            arguments_few_shots_samples_df_list.append(df)

    arguments_few_shots = pd.concat(arguments_few_shots_samples_df_list)

    if few_shot_setting.sort_random:
        arguments_few_shots = arguments_few_shots.sample(frac=1, random_state=random_seed)
    else:
        arguments_few_shots = arguments_few_shots.sort_values(by=few_shot_setting.sort_values_by, 
                                                              ascending=few_shot_setting.sort_values_ascending)

    return arguments_few_shots

def make_instruction_few_shots_row(instruction_row, arguments_few_shots):
    instruction_language = instruction_row["Instruction-Language"]
    instruction_is_formal = instruction_row["Is-Formal"]
    instruction = instruction_row["Instruction"]

    if instruction_language == "en":
        instruction_1 = f"{instruction[0].lower()}{instruction[1:]}"
    else:
        instruction_1 = instruction

    arguments_condition = (arguments_few_shots["Pattern-Language"] == instruction_language) & \
                            (arguments_few_shots["Is-Formal"] == instruction_is_formal)
    arguments_view = arguments_few_shots[arguments_condition].reset_index(drop=True)
    
    example_arguments = ""
    for idx, argument_row in arguments_view.iterrows():
        argument = argument_row["Argument"]
        label = "Valid" if argument_row["Is-Valid"] == 1 else "Invalid"
        
        if idx == 0:
            example_arguments = f"{argument}: {label};"
        else:
            example_arguments = f"{example_arguments} {argument}: {label};"
    
    template_few_shots = TEMPLATES_FEW_SHOTS[1][instruction_language]

    instruction_few_shots = template_few_shots.replace("<INSTRUCTION_0>", instruction)
    instruction_few_shots = instruction_few_shots.replace("<EXAMPLE_ARGUMENTS>", example_arguments)
    instruction_few_shots = instruction_few_shots.replace("<INSTRUCTION_1>", instruction_1)
    instruction_few_shots = instruction_few_shots.replace("\n", " ")

    instruction_row["Instruction-ID"] += "-few-shots"
    instruction_row["Instruction"] = instruction_few_shots

    return instruction_row

def generate_instructions_few_shots(instructions_base, arguments_few_shots):
    instructions_few_shots = instructions_base.apply(make_instruction_few_shots_row, axis=1, arguments_few_shots=arguments_few_shots)

    return instructions_few_shots.reset_index(drop=True)

def generate_arguments_and_instructions_few_shots(arguments_all, instructions_base, few_shots_settings_list, random_seed):
    arguments_few_shots_df_list = []
    instructions_few_shots_df_list = []
    for idx, few_shot_setting in enumerate(few_shots_settings_list):
        arguments_few_shots = generate_arguments_few_shots(arguments_all, few_shot_setting, random_seed)
        instructions_few_shots = generate_instructions_few_shots(instructions_base, arguments_few_shots)

        arguments_few_shots_df_list.append(arguments_few_shots)
        instructions_few_shots_df_list.append(instructions_few_shots)

    return arguments_few_shots_df_list, instructions_few_shots_df_list

def generate_prompts(instructions_all, arguments_effective, few_shots_setting_idx):
    prompts = pd.merge(instructions_all, arguments_effective, how="cross")
    prompts = prompts[prompts["Is-Formal_x"] == prompts["Is-Formal_y"]].drop(columns=["Is-Formal_y"])
    prompts = prompts.rename(columns={
        "Is-Formal_x": "Is-Formal",
    })
    prompts["Prompt"] = prompts.apply(lambda row: f"{row['Instruction']}: {row['Argument']}", axis=1)
    # prompts["Prompt"] = \
    #     prompts.apply(lambda row: row["Prompt"].replace("PARTY", 
    #                                                     POLITICS_DICT[row["Instruction-Language"]][row["Political-Leaning"]]["PARTY"]), axis=1)
    prompts["Is-Few-Shots"] = prompts.apply(lambda row: 1 if "-few-shots" in row["Instruction-ID"] else 0, axis=1)
    prompts["Few-Shots-Setting-Idx"] = few_shots_setting_idx
    prompts = prompts[PROMPTS_COLUMNS]

    return prompts.reset_index(drop=True)

def row_present_in_df(row, df):
    return ((row["Gender"]==df["Gender"]) & (row["Is-Political"]==df["Is-Political"]) & \
            (row["Is-Formal"]==df["Is-Formal"]) & (row["Pattern-Language"]==df["Pattern-Language"]) & \
            (row["Pattern-ID"]==df["Pattern-ID"]) & (row["Political-Leaning"]==df["Political-Leaning"]) & \
            (row["Is-Valid"]==df["Is-Valid"]) & (row["Variation-ID"]==df["Variation-ID"])).any()

sort_values_by = [
    "Pattern-Language",
    "Is-Political",
    "Is-Formal",
    "Pattern-ID",
    "Gender",
    "Political-Leaning",
    "Is-Valid",
    "Variation-ID",
    # "Instruction-ID",
    # "Is-Few-Shots",
    # "Few-Shots-Setting-Idx",
]
sort_values_ascending = [
    False, # "Pattern-Language",
    False, # "Is-Political",
    False, # "Is-Formal",
    True, # "Pattern-ID",
    True, # "Gender",
    True, # "Political-Leaning",
    True, # "Is-Valid",
    True, # "Variation-ID",
    # True, # "Instruction-ID",
    # True, # "Is-Few-Shots",
    # True, # "Few-Shots-Setting-Idx",
]

def generate_prompts_all(instructions_base, arguments_all, arguments_few_shots_df_list, instructions_few_shots_df_list, few_shots_settings_list):
    idx_list = list(range(len(few_shots_settings_list)+1))
    arguments_few_shots_df_list = [pd.DataFrame(columns=ARGUMENTS_COLUMNS)] + arguments_few_shots_df_list
    instructions_few_shots_df_list = [instructions_base] + instructions_few_shots_df_list

    prompts_df_list = []

    for idx, arguments_few_shots, instructions_few_shots in zip(idx_list, arguments_few_shots_df_list, instructions_few_shots_df_list):
        prompts = generate_prompts(instructions_few_shots, arguments_all, idx)
        prompts["Argument-Is-In-Few-Shots"] = \
            prompts.apply(lambda row: 1 if row_present_in_df(row, arguments_few_shots) else 0, axis=1)
        prompts_df_list.append(prompts)

    prompts_all = pd.concat(prompts_df_list)
    prompts_all = prompts_all[prompts_all["Argument-Is-In-Few-Shots"]==0][PROMPTS_COLUMNS]
    prompts_all = prompts_all.sort_values(by=sort_values_by, ascending=sort_values_ascending).reset_index(drop=True)

    return prompts_all

def generate_prompts_same_language_cross_language(prompts, prompts_path):
    prompts_same_language = prompts[prompts["Pattern-Language"] == prompts["Instruction-Language"]]
    prompts_cross_language = prompts[prompts["Pattern-Language"] != prompts["Instruction-Language"]]

    prompts_same_language_path = prompts_path
    prompts_same_language.to_csv(prompts_same_language_path.resolve(), index=False)

    return prompts_same_language.reset_index(drop=True), prompts_cross_language.reset_index(drop=True)

def construct_results_header_str():
    results_header_str = f"{RESULTS_COLUMNS[0]}"
    
    for column_name in RESULTS_COLUMNS[1:]:
        results_header_str = f"{results_header_str},{column_name}"
    
    return results_header_str

def extract_grade(generated_output):
    extracted_grade=""

    onlynumberpattern=re.search(r'^\s*([0-9]+\.*,*[0-9]*)\s*$', generated_output)
    negatedpattern=re.search(r'(\*\*\s*)?(nicht|not|NOT)(\*\*\s*)? (deductively'
                             '|deduktiv|logisch|logically)\s*([a-zA-Zäöü]+)\s*',
                             generated_output)
    ungueltigpattern=re.search(r'(does not deductively follow|does not follow'
                               'deductively|ist ungültig|ist nicht gültig'
                               '|Ungültig|folgt nicht (formal-deduktiv )?'
                               'aus den Prämissen'
                               '|nicht (logisch|notwendigerweise) gültig'
                               '|nicht formal-deduktiv gültig'
                               '|folgt nicht formal-deduktiv'
                               '|folgt formal-deduktiv nicht'
                               '|als ungültig betrachtet'
                               '|als ungültig bezeichnet'
                               '|Schluss ungültig'
                               '|ungültigen Schluss)',
                               generated_output)
    invalidpattern=re.search(r'(is(?:t) invalid|is not valid|not deductively'
                             ' valid|Invalid|nicht deduktiv-formal valid)'
                             '|is invalid'
                             '|structurally invalid'
                             '|not (completely)? materially valid'
                             '|not (completely)? deductive-material valid'
                             '|not (completely)? deductively-material valid'
                             '|not (completely)? deductive-materially valid'
                             '|not (completely)? deductively-materially valid'
                             '|not (completely)? appear to be (deductively|necessarily|logically)? valid',
                             generated_output)
    deductivelypattern=re.search(r'(is|ist|appears) (deductively|deduktiv|'
    'logisch|logically)\s*([a-zA-Z]+)\s*', generated_output)
    gueltigpattern=re.search(r'(ist gültig|ist tatsächlich gültig|Gültig|'
                               ' (logisch|deduktiv) gültig)', generated_output)
    validpattern=re.search(r'(is valid|(is|argument) (indeed)?'
                           '(deductively|logically|materially|material|'
                           'material-deductive|deductively-material)'
                             ' valid|Valid)'
                             '|(is|appears to be|seems) structurally valid'
                             '|(is|appears to be|seems) deductive-material valid'
                             '|(is|appears to be|seems) deductively-material valid'
                             '|(is|appears to be|seems) deductive-materially valid'
                             '|(is|appears to be|seems) deductively-materially valid', generated_output)
    logicallyfollows=re.search(r'(conclusion logically follows|conclusion '
                               'follows logically)', generated_output)
    numberstarspattern=re.search(r'\*\*\s*([0-9]+\.*,*[0-9]*)\s*\*\*$',
                                 generated_output)
    beginningpattern=re.search(r'^\s*([0-9]+\.*,*[0-9]*|valid|invalid|'
                                    'gültig|ungültig)\s*',
                                    generated_output, re.IGNORECASE)
    npattern=re.search(r'^\s*N\s*([0-9]+\.*,*[0-9]*)\s*', generated_output)
    asapattern=re.search(r'\s*(a|as|as a|mit|a rating of|rate this argument'
                         '|einen Wert von|einer? Note von|dieses Argument mit'
                         '|das Argument auf eine|Durchschnittswert'
                         '|diese Argumentation als'
                         '|dieses Argument mit einer|diesem Argument eine'
                         '|wie folgt:|einer? Bewertung von|folgt bewerten:'
                         '|dem Argument eine|mit einer|mit einem Wert von'
                         '|this argument rates|beurteilen:|folgendermassen:'
                         '|a score of)'
                         '\s*([0-9]+\.*,*[0-9]*)\s*',
                         generated_output)
    nichtkorrekt=re.search(r"(Deduktion|Schluss|Schlussfolgerung)? ist nicht "
                           "(zwingend |ganz |notwendigerweise)?(gültig|korrekt)",generated_output)
    korrekt=re.search("(Deduktion|Schluss|Schlussfolgerung) (ist|scheint) "
                      "(zwingend |ganz )?(gültig|korrekt)",generated_output)

    graderatingpattern=re.search(r'(Rating|Grade|Score|Note|Punktzahl):'
                                 '\s*([0-9]+\.*\,*[0-9]*)', generated_output)
    gemmagrading=re.search(r'\*\*\s*([0-9]+\.*\,*[0-9]*) (out of|of|von) 5'
                           '\s*\*\*', generated_output)
    starspattern=re.search(r'\*\*\s*([a-zA-Zäöü]+?|[0-9]+\.*,*[0-9]*|'
                           'deduktiv gültig|deductively valid)\s*\*\*',
                           generated_output)

    if onlynumberpattern:
        extracted_grade =onlynumberpattern.group(1).lower().rstrip(" ")
        return float(extracted_grade.replace(",","."))
    elif negatedpattern or invalidpattern or ungueltigpattern or nichtkorrekt:
        extracted_grade = 0
        return extracted_grade
    elif gueltigpattern or validpattern or logicallyfollows or korrekt:
        extracted_grade = 1
        return extracted_grade
    elif numberstarspattern:
        extracted_grade =numberstarspattern.group(1).lower().rstrip(" ")
        return float(extracted_grade)
    elif gemmagrading:
        extracted_grade =gemmagrading.group(1).lower().rstrip(" ")
        return float(extracted_grade)
    elif beginningpattern:
        extracted_grade =beginningpattern.group(1).lower().rstrip(" ")
        return extracted_grade.strip( )
    elif npattern:
        extracted_grade =npattern.group(1).lower().rstrip(" ")
        return float(extracted_grade)
    elif asapattern:
        extracted_grade =asapattern.group(2).lower().rstrip(" ")
        return extracted_grade
    elif deductivelypattern:
        extracted_grade =deductivelypattern.group(3).lower().rstrip(" ")
        extracted_grade_clean=1
        return extracted_grade_clean

    elif graderatingpattern:
        extracted_grade =graderatingpattern.group(2).lower().rstrip(" ")
        return float(extracted_grade.replace(",","."))
    elif starspattern:
        extracted_grade =starspattern.group(1).lower().rstrip(" ")
        return extracted_grade.strip( )

    else:
        extracted_grade="NOMATCH"
        return extracted_grade
    
def post_processing_extracted_grade(bare_grade):
    result = None
    
    if not isinstance(bare_grade, str):
        result = bare_grade
    else:
        stripped_bare_grade=bare_grade.strip()

        # Now begin a number of cleaning matches
        if re.search(r'^([0-9]+\.*,*[0-9]*)/',stripped_bare_grade):
            temp_match=re.search(r'^([0-9]+\.*,*[0-9]*)/',stripped_bare_grade)
            result=temp_match.group(1).lower().rstrip(" ")
        elif re.search(r'^[0-9]',stripped_bare_grade):
            #print("Starting conversion with",current_bare_grade,file=sys.stderr)
            result=float(stripped_bare_grade.replace(",","."))
            #print("Converted:",current_bare_grade,file=sys.stderr)

        elif re.search(r'^(valid|(deduktiv)? gültig|korrekt|gültig'
                       '|deductively valid)',
                       stripped_bare_grade):
            result=1
        elif re.search(r'^(invalid|ungültig|unzulässig)',
                       stripped_bare_grade):
            result=0
        elif re.search(r'^([0-9])',stripped_bare_grade):
            temp_match=re.search(r'^([0-9])',stripped_bare_grade)
            result=float(temp_match.group(1).lower().rstrip(" "))
        else:
            print("Failed to convert this bare grade, skipping:",
                  stripped_bare_grade,file=sys.stderr)

        return result

    
def sanitize_string_for_csv(content):
    result = content.replace('"', '""')

    if '\n' in result or ',' in result or '"' in result:
        result = f'"{result}"'

    return result


def run_model_open_ai(prompts, results_path, model_name):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ["OPENAI_API_KEY"],
    )

    results_header_str = construct_results_header_str()
    output_file = open(results_path,"w")
    print(results_header_str, file=output_file)

    for idx, prompt_row in prompts.iterrows():
        messages = [{
            "role": "user",
            "content": prompt_row["Prompt"]
        }]
        
        output = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        
        output_response = output.choices[0].message.content
        sanitized_prompt = sanitize_string_for_csv(prompt_row["Prompt"])
        sanitized_output_response = sanitize_string_for_csv(output_response)

        extracted_grade = extract_grade(output_response)

        output_grade = extracted_grade

        now = datetime.now()
        created_at = now.strftime("%d/%m/%Y, %H:%M:%S")

        output_row_str = f"{prompt_row['Gender']},{prompt_row['Is-Political']},"\
            f"{prompt_row['Is-Formal']},{prompt_row['Pattern-Language']},{prompt_row['Pattern-ID']},"\
            f"{prompt_row['Political-Leaning']},{prompt_row['Is-Valid']},{prompt_row['Variation-ID']}|"\
            f"{prompt_row['Instruction-Language']},{prompt_row['Instruction-ID']},{prompt_row['Is-Few-Shots']},"\
            f"{prompt_row['Few-Shots-Setting-Idx']},{sanitized_prompt},"\
            f"{created_at},{model_name},{sanitized_output_response},{output_grade}"

        print(output_row_str, file=output_file)
        
        if idx % 5 == 0:
            output_file.flush()
        
    output_file.close()

def run_model_llama_cpp(prompts, results_path, model_name, model_path):
    model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=6144,
        verbose=True
    )

    results_header_str = construct_results_header_str()
    output_file = open(results_path,"w")
    print(results_header_str, file=output_file)

    for idx, prompt_row in prompts.iterrows():
        messages = [{
            "role": "user",
            "content": prompt_row["Prompt"]
        }]
        
        output_dict = model.create_chat_completion(messages=messages)
        
        output_response = output_dict['choices'][0]['message']['content']
        sanitized_prompt = sanitize_string_for_csv(prompt_row["Prompt"])
        sanitized_output_response = sanitize_string_for_csv(output_response)
        
        extracted_grade = extract_grade(output_response)

        output_grade = extracted_grade

        now = datetime.now()
        created_at = now.strftime("%d/%m/%Y, %H:%M:%S")

        output_row_str = f"{prompt_row['Gender']},{prompt_row['Is-Political']},"\
            f"{prompt_row['Is-Formal']},{prompt_row['Pattern-Language']},{prompt_row['Pattern-ID']},"\
            f"{prompt_row['Political-Leaning']},{prompt_row['Is-Valid']},{prompt_row['Variation-ID']},"\
            f"{prompt_row['Instruction-Language']},{prompt_row['Instruction-ID']},{prompt_row['Is-Few-Shots']},"\
            f"{prompt_row['Few-Shots-Setting-Idx']},{sanitized_prompt},"\
            f"{created_at},{model_name},{sanitized_output_response},{output_grade}"

        print(output_row_str, file=output_file)
        
        if idx % 5 == 0:
            output_file.flush()
        
    output_file.close()

def post_process_output(results_row):
    response = results_row["Output"]
    extracted_grade = extract_grade(response)
    bare_grade = post_processing_extracted_grade(extracted_grade)

    if bare_grade is None:
        grade = extracted_grade
    else:
        grade = bare_grade

    results_row["Bare-Grade"] = grade
    results_row["Failure"] = isinstance(grade, str) or (grade != 0 and grade != 1)
    
    return results_row