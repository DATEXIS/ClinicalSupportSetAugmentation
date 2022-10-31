import glob
import json
import os
import pickle
from collections import defaultdict
from csv import DictReader
from typing import Dict, Union, List, Set, Optional

import psycopg2
from psycopg2.extras import DictCursor

from src.UMLSParser.UMLSParser import UMLSParser
import src.config as config
from src.utils import ICDDescriptionCache, WikidataDescriptionCache, CCSDXMapper


def init_database_cursor() -> psycopg2.extras.DictCursor:
    conn = psycopg2.connect(host=config.umls_db_host, database=config.umls_db_name, user=config.umls_db_user, password=config.umls_db_pass)
    return conn.cursor(cursor_factory=DictCursor)


def get_diagnoses_codes_for_hadm_ids(cur: psycopg2.extras.DictCursor) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = defaultdict(list)
    cur.execute('SELECT hadm_id, icd9_code FROM mimiciii.diagnoses_icd')
    for row in cur.fetchall():
        result[str(row['hadm_id'])].append(row['icd9_code'])
    return result


def to_3_digit_code(code: str):
    if code[0].lower() == 'v' or code[0].lower() == 'e':
        return code[0:4]
    else:
        return code[0:3]


if __name__ == '__main__':
    cur = init_database_cursor()
    umls = UMLSParser(config.umls_path, language_filter=['ENG'])
    icd_description_cache = ICDDescriptionCache(umls, cur)
    umls_matcher = icd_description_cache.umls_matcher
    karl_wikidata = WikidataDescriptionCache(config.wikidata_extract_path, ignore_ambigous_matches=True)
    ccs_dx_mapper = CCSDXMapper(path_to_ccs_dx_folder=config.path_to_ccs_dx_folder)

    additional_definitions_from_wikidata = 0
    data_splits: Dict[str, Dict[str, Dict[str, Union[str, List[str], Set[str]]]]] = defaultdict(dict)
    icd9_codes_by_hadm_id = get_diagnoses_codes_for_hadm_ids(cur)
    distinct_short_codes_per_split = defaultdict(set)
    distinct_icd9_codes = set()
    distinct_hadm_ids = set()

    for cop_file in glob.glob(os.path.join(config.path_to_van_aken_dataset, '*.csv')):
        with open(cop_file, 'r') as cop_in:
            split = os.path.basename(cop_file).split('_')[-1].split('.')[0]
            csvreader = DictReader(cop_in)
            for cop_row in csvreader:
                hadm_id = cop_row['id']
                distinct_hadm_ids.add(hadm_id)

                bettys_codes = set(cop_row['short_codes'].split(','))
                toms_codes = set([to_3_digit_code(x) for x in icd9_codes_by_hadm_id[hadm_id]])
                ccs_codes = set(
                    [ccs_dx_mapper.get_ccs_category_for_icd9_code(x) for x in icd9_codes_by_hadm_id[hadm_id]])

                assert (bettys_codes == toms_codes)

                short_code_definitions = defaultdict(lambda: defaultdict(list))
                ccs_code_definitions = defaultdict(lambda: defaultdict(list))
                ccs_code_names = {}
                for icd9_code in icd9_codes_by_hadm_id[hadm_id]:
                    distinct_icd9_codes.add(icd9_code)
                    short_code = to_3_digit_code(icd9_code)
                    ccs_code = ccs_dx_mapper.get_ccs_category_for_icd9_code(icd9_code)
                    distinct_short_codes_per_split[split].add(short_code)
                    ccs_code_names[ccs_code] = ccs_dx_mapper.get_description_for_css_category(ccs_code)
                    short_name, long_name, definition = icd_description_cache.get_umls_description_for_mimic_icd9_diagnosis_label(
                        icd9_code)
                    if short_name:
                        short_code_definitions[short_code]['short_names'].append(short_name)
                        ccs_code_definitions[ccs_code]['short_names'].append(short_name)
                    else:
                        short_code_definitions[short_code]['short_names'].append('')
                        ccs_code_definitions[ccs_code]['short_names'].append('')
                    if long_name:
                        short_code_definitions[short_code]['long_names'].append(long_name)
                        ccs_code_definitions[ccs_code]['long_names'].append(long_name)
                    else:
                        short_code_definitions[short_code]['long_names'].append('')
                        ccs_code_definitions[ccs_code]['long_names'].append('')
                    if definition:
                        short_code_definitions[short_code]['definitions'].append(definition)
                        ccs_code_definitions[ccs_code]['definitions'].append(definition)
                    elif karl_wikidata.get_definition_for_mimic3_diagnosis_label(icd9_code):
                        short_code_definitions[short_code]['definitions'].append(
                            karl_wikidata.get_definition_for_mimic3_diagnosis_label(icd9_code)[0])
                        ccs_code_definitions[ccs_code]['definitions'].append(
                            karl_wikidata.get_definition_for_mimic3_diagnosis_label(icd9_code)[0])
                        additional_definitions_from_wikidata += 1
                    else:
                        short_code_definitions[short_code]['definitions'].append('')
                        ccs_code_definitions[ccs_code]['definitions'].append('')

                data_splits[split][hadm_id] = {
                    'hadm_id': hadm_id,
                    'admission_note': cop_row['text'],
                    'short_codes': list(bettys_codes),
                    'ccs_codes': list(ccs_codes),
                    'mimic_icd9_codes': icd9_codes_by_hadm_id[hadm_id],
                    'short_code_definitions': short_code_definitions,
                    'ccs_code_definitions': ccs_code_definitions,
                    'ccs_code_names': ccs_code_names
                }

    print(', '.join([f'{k}: {len(v)}' for k, v in distinct_short_codes_per_split.items()]), len(distinct_hadm_ids))

    num_icd9_codes = len(distinct_icd9_codes)
    num_icd9_short_names = 0
    num_icd9_long_names = 0
    num_icd9_definitions = 0
    num_icd9_wikidata_definitions = 0
    num_icd9_additional_wikidata_definitions = 0
    for icd9_code in distinct_icd9_codes:
        short_name, long_name, definition = icd_description_cache.get_umls_description_for_mimic_icd9_diagnosis_label(
            icd9_code)
        wikidata = karl_wikidata.get_definition_for_mimic3_diagnosis_label(icd9_code)
        if short_name is not None:
            num_icd9_short_names += 1
        if long_name is not None:
            num_icd9_long_names += 1
        if definition is not None:
            num_icd9_definitions += 1
        if wikidata is None or len(wikidata) != 0:
            num_icd9_wikidata_definitions += 1
        if wikidata is not None and len(wikidata) != 0 and definition is None:
            num_icd9_additional_wikidata_definitions += 1

    print(
        f'num_icd_codes: {num_icd9_codes}, num_icd9_short_names: {num_icd9_short_names} num_icd9_long_names: {num_icd9_long_names} num_icd9_definitions:{num_icd9_definitions} num_icd9_additional_wikidata_definitions:{num_icd9_additional_wikidata_definitions}')

    for split, data in data_splits.items():
        with open(config.path_to_out_data.format(split), 'w') as data_file, \
                open(config.path_to_out_labels.format(split), 'w') as label_file, \
                open(config.path_to_out_umls_wikidata_labels.format(split), 'w') as extended_label_file, \
                open(config.path_to_out_ccs_labels.format(split), 'w') as ccs_label_file:
            data_list = []

            for hadm_id, hadm_data in data.items():

                for short_code in hadm_data['short_codes']:
                    label_file.write(f'{hadm_id},{short_code}\n')

                for ccs_code in hadm_data['ccs_codes']:
                    ccs_label_file.write(f'{hadm_id},{ccs_code}\n')

                data_list.append({
                    'id': int(hadm_id),
                    'text': hadm_data['admission_note']
                })
                extended_label_file.write(json.dumps(hadm_data) + '\n')

            json.dump(data_list, data_file)
