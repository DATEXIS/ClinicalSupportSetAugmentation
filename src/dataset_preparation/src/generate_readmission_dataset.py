import datetime
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union, Set, Optional

import psycopg2
from psycopg2.extras import DictCursor
from tqdm import tqdm

import config
from src.UMLSParser.UMLSParser import UMLSParser
from src.UMLSParser.model import Concept
from utils import MIMIC2UMLSMatcher, CCSDXMapper, ICDDescriptionCache

@dataclass
class DischargeSummary:
    description: str
    text: str
    chartdate: datetime.datetime


@dataclass
class Admission:
    adm_id: int
    adm_date: datetime.datetime
    discharge_summaries: List[DischargeSummary]
    diagnoses: list
    diagnoses_umls: List[Concept]
    procedures: list
    procedures_umls: List[Concept]


def init_database_cursor() -> psycopg2.extras.DictCursor:
    conn = psycopg2.connect(host=config.umls_db_host, database=config.umls_db_name, user=config.umls_db_user, password=config.umls_db_pass)
    return conn.cursor(cursor_factory=DictCursor)


def get_discharge_summaries_by_admission(cur: psycopg2.extras.DictCursor) -> Dict[str, List[DischargeSummary]]:
    discharge_summaries_by_admission: Dict[str, List[DischargeSummary]] = defaultdict(list)

    cur.execute("SELECT admissions.hadm_id, chartdate, description, text FROM mimiciii.admissions "
                "INNER JOIN mimiciii.noteevents ON noteevents.hadm_id = admissions.hadm_id "
                "WHERE noteevents.category = 'Discharge summary' "
                "ORDER BY admissions.hadm_id, chartdate")

    for row in tqdm(cur.fetchall(), desc="Parsing discharge summaries"):
        discharge_summaries_by_admission[row['hadm_id']].append(
            DischargeSummary(description=row['description'], text=row['text'], chartdate=row['chartdate']))

    return discharge_summaries_by_admission


def get_admissions_by_patient(cur: psycopg2.extras.DictCursor, umls_matcher: MIMIC2UMLSMatcher) -> Dict[
    int, List[Admission]]:
    """
    Returns ordered patient admissions for patients with multiple admissions.
    :param cur:
    :param umls_concepts_by_icd9_id:
    :return:
    """

    admissions_by_patient = defaultdict(list)
    __readmissions_by_hadm_id: Dict[int, Admission] = dict()
    __discharge_summaries_by_admission = get_discharge_summaries_by_admission(cur)

    cur.execute(
        "SELECT admissions.row_id, admissions.hadm_id, admissions.subject_id, admissions.admittime as adm_date "
        "FROM mimiciii.admissions "
        "WHERE admissions.subject_id IN "
        "(SELECT mimiciii.admissions.subject_id FROM mimiciii.admissions GROUP BY admissions.subject_id "
        "HAVING count(admissions.row_id) > 1) ORDER BY admissions.subject_id, admittime")

    for row in tqdm(cur.fetchall(), desc="Parsing admissions"):
        admission = Admission(adm_id=row['hadm_id'], adm_date=row['adm_date'],
                              discharge_summaries=__discharge_summaries_by_admission[row['hadm_id']], diagnoses=[],
                              procedures=[], diagnoses_umls=[],
                              procedures_umls=[])
        admissions_by_patient[row['subject_id']].append(admission)
        __readmissions_by_hadm_id[row['hadm_id']] = admission

    cur.execute("SELECT hadm_id, subject_id, icd9_code FROM mimiciii.diagnoses_icd WHERE diagnoses_icd.icd9_code IS NOT NULL")
    for row in cur.fetchall():
        if row['hadm_id'] not in __readmissions_by_hadm_id.keys():
            continue
        __readmissions_by_hadm_id[row['hadm_id']].diagnoses.append(row['icd9_code'])
        __readmissions_by_hadm_id[row['hadm_id']].diagnoses_umls.append(
            umls_matcher.get_concept_for_diagnosis(row['icd9_code']))

    cur.execute("SELECT hadm_id, subject_id, icd9_code FROM mimiciii.procedures_icd WHERE procedures_icd.icd9_code IS NOT NULL")
    for row in cur.fetchall():
        if row['hadm_id'] not in __readmissions_by_hadm_id.keys():
            continue
        __readmissions_by_hadm_id[row['hadm_id']].procedures.append(row['icd9_code'])
        __readmissions_by_hadm_id[row['hadm_id']].procedures_umls.append(
            umls_matcher.get_concept_for_procedure(row['icd9_code']))

    return admissions_by_patient


def concepts_to_dicts(concepts: Set[Concept], umls: UMLSParser) -> List[Union[None, Dict[str, Union[str, List[str]]]]]:
    """
    Concepts are not serializable, so we use this to define the JSON rendering.
    :param umls: UMLSParser
    :param concepts:
    :return:
    """

    result = []
    for concept in concepts:

        if concept is None:
            result.append(None)
            continue

        result.append({
            'cui': concept.get_cui(),
            'tui': concept.get_tui(),
            'semantic_type': umls.get_semantic_types()[concept.get_tui()].get_name(),
            'icd9_ids': list(concept.get_source_ids().get('ICD9CM')),
            'name': concept.get_preferred_names_for_language('ENG')[0],
            'definitions': list(concept.get_definitions())
        })
    return result


def get_season_from_datetime(date: datetime.datetime) -> str:
    leapyear_offset = datetime.datetime(date.timetuple().tm_year, 12, 31).timetuple().tm_yday - 365
    day_of_year = date.timetuple().tm_yday - leapyear_offset

    if 0 <= day_of_year < 79:
        return 'winter'
    if 79 <= day_of_year < 172:
        return 'spring'
    if 172 <= day_of_year < 266:
        return 'summer'
    if 266 <= day_of_year < 355:
        return 'fall'
    if day_of_year >= 355:
        return 'winter'
    raise Exception('Season decoder error')


def to_3_digit_code(code: Union[str, None]) -> Optional[str]:
    if code is None:
        return code
    if code[0].lower() == 'v' or code[0].lower() == 'e':
        return code[0:4]
    else:
        return code[0:3]


def collate_umls_set_to_3_digit_code_dict(concepts: Set[Concept], umls: UMLSParser) -> Dict[str, List[Union[None, Dict[str, Union[str, List[str]]]]]]:
    tmp = defaultdict(set)
    for concept in concepts:
        if not concept:
            continue
        icd9_codes = concept.get_source_ids()['ICD9CM']
        for icd9_code in icd9_codes:
            mimic_code = icd9_code.replace('.', '')
            short_code = to_3_digit_code(mimic_code)
            tmp[short_code].add(concept)
    ret_val = dict()
    for short_code, concept_set in tmp.items():
        ret_val[short_code] = concepts_to_dicts(concept_set, umls)
    return ret_val


if __name__ == '__main__':
    result = []
    cur = init_database_cursor()
    umls = UMLSParser(config.umls_path, language_filter=['ENG'])
    umls_matcher = MIMIC2UMLSMatcher(umls)
    ccs_dx_mapper = CCSDXMapper(path_to_ccs_dx_folder=config.path_to_ccs_dx_folder)
    icd_description_cache = ICDDescriptionCache(umls, cur)

    for subject_id, admissions in tqdm(get_admissions_by_patient(cur=cur, umls_matcher=umls_matcher).items(),
                                       desc='Aggregating re-admissions'):

        patient_history = []

        if len(admissions) < 2:
            raise Exception('Found a patient with only one admission. This should not be the case.')
        for i in range(len(admissions)):
            first_adm: Admission = admissions[0]
            current_adm: Admission = admissions[i]
            days_since_first_admission = (current_adm.adm_date - first_adm.adm_date).days

            if i == 0:
                new_diagnoses_umls: set = set(current_adm.diagnoses_umls)
                lost_diagnoses_umls: set = set()
                persistent_diagnoses_umls: set = set()

                new_diagnoses_short_code_umls: Dict[str, List[Union[None, Dict[str, Union[str, List[str]]]]]] = collate_umls_set_to_3_digit_code_dict(new_diagnoses_umls, umls)
                lost_diagnoses_short_code_umls: Optional[Dict[str, List[Union[None, Dict[str, Union[str, List[str]]]]]]] = None
                persistent_diagnoses_short_code_umls: Optional[Dict[str, List[Union[None, Dict[str, Union[str, List[str]]]]]]] = None

                new_diagnoses: set = set(current_adm.diagnoses)
                lost_diagnoses: set = set()
                persistent_diagnoses: set = set()

                new_diagnoses_short_code: set = set([to_3_digit_code(x) for x in current_adm.diagnoses])
                lost_diagnoses_short_code: set = set()
                persistent_diagnoses_short_code: set = set()

                new_diagnoses_ccs: set = set([ccs_dx_mapper.get_ccs_category_for_icd9_code(x) for x in current_adm.diagnoses])
                lost_diagnoses_ccs: set = set()
                persistent_diagnoses_ccs: set = set()

                new_diagnoses_ccs_descriptions = dict()
                for icd9_code in new_diagnoses:
                    ccs_code = ccs_dx_mapper.get_ccs_category_for_icd9_code(icd9_code)
                    short_name, long_name, definition = icd_description_cache.get_umls_description_for_mimic_icd9_diagnosis_label(icd9_code)
                    definition = ['' if definition is None else definition]
                    new_diagnoses_ccs_descriptions[ccs_code] = {
                        'short_name': short_name,
                        'long_name': long_name,
                        'definition': definition
                    }

                lost_diagnoses_ccs_descriptions = dict()
                persistent_diagnoses_ccs_descriptions = dict()

                assert len(new_diagnoses_ccs) == len(new_diagnoses_ccs_descriptions)

            else:
                _last_diagnoses_umls: set = set(admissions[i - 1].diagnoses_umls)
                _current_diagnoses_umls: set = set(current_adm.diagnoses_umls)
                new_diagnoses_umls: set = _current_diagnoses_umls - _last_diagnoses_umls
                lost_diagnoses_umls: set = _last_diagnoses_umls - _current_diagnoses_umls
                persistent_diagnoses_umls: set = _current_diagnoses_umls & _last_diagnoses_umls

                _last_diagnoses_short_code_umls: Dict[str, List[Union[None, Dict[str, Union[str, List[str]]]]]] = collate_umls_set_to_3_digit_code_dict(_last_diagnoses_umls, umls)
                _current_diagnoses_short_code_umls: Dict[str, List[Union[None, Dict[str, Union[str, List[str]]]]]] = collate_umls_set_to_3_digit_code_dict(_current_diagnoses_umls, umls)
                new_diagnoses_short_code_umls: Dict[str, List[Union[None, Dict[str, Union[str, List[str]]]]]] = dict([(x, y) for x,y in _current_diagnoses_short_code_umls.items() if x not in _last_diagnoses_short_code_umls.keys()])
                lost_diagnoses_short_code_umls: Dict[str, List[Union[None, Dict[str, Union[str, List[str]]]]]] = dict([(x, y) for x,y in _last_diagnoses_short_code_umls.items() if x not in _current_diagnoses_short_code_umls.keys()])
                persistent_diagnoses_short_code_umls: Dict[str, List[Union[None, Dict[str, Union[str, List[str]]]]]] = dict([(x, y) for x,y in _current_diagnoses_short_code_umls.items() if x in _last_diagnoses_short_code_umls.keys()])

                _last_diagnoses: set = set(admissions[i - 1].diagnoses)
                _current_diagnoses: set = set(current_adm.diagnoses)
                new_diagnoses: set = _current_diagnoses - _last_diagnoses
                lost_diagnoses: set = _last_diagnoses - _current_diagnoses
                persistent_diagnoses: set = _current_diagnoses & _last_diagnoses

                _last_diagnoses_short_code: set = set([to_3_digit_code(x) for x in admissions[i - 1].diagnoses])
                _current_diagnoses_short_code: set = set([to_3_digit_code(x) for x in current_adm.diagnoses])
                new_diagnoses_short_code: set = _current_diagnoses_short_code - _last_diagnoses_short_code
                lost_diagnoses_short_code: set = _last_diagnoses_short_code - _current_diagnoses_short_code
                persistent_diagnoses_short_code: set = _current_diagnoses_short_code & _last_diagnoses_short_code

                _last_diagnoses_ccs: set = set([ccs_dx_mapper.get_ccs_category_for_icd9_code(x) for x in admissions[i - 1].diagnoses])
                _current_diagnoses_ccs: set = set([ccs_dx_mapper.get_ccs_category_for_icd9_code(x) for x in current_adm.diagnoses])
                new_diagnoses_ccs: set = _current_diagnoses_ccs - _last_diagnoses_ccs
                lost_diagnoses_ccs: set = _last_diagnoses_ccs - _current_diagnoses_ccs
                persistent_diagnoses_ccs: set = _current_diagnoses_ccs & _last_diagnoses_ccs

                new_diagnoses_ccs_descriptions = dict()
                for icd9_code in new_diagnoses:
                    ccs_code = ccs_dx_mapper.get_ccs_category_for_icd9_code(icd9_code)
                    if ccs_code not in new_diagnoses_ccs:
                        continue
                    short_name, long_name, definition = icd_description_cache.get_umls_description_for_mimic_icd9_diagnosis_label(icd9_code)
                    definition = ['' if definition is None else definition]
                    new_diagnoses_ccs_descriptions[ccs_code] = {
                        'short_name': short_name,
                        'long_name': long_name,
                        'definition': definition
                    }

                lost_diagnoses_ccs_descriptions = dict()
                for icd9_code in lost_diagnoses:
                    ccs_code = ccs_dx_mapper.get_ccs_category_for_icd9_code(icd9_code)
                    if ccs_code not in lost_diagnoses_ccs:
                        continue
                    short_name, long_name, definition = icd_description_cache.get_umls_description_for_mimic_icd9_diagnosis_label(icd9_code)
                    definition = ['' if definition is None else definition]
                    lost_diagnoses_ccs_descriptions[ccs_code] = {
                        'short_name': short_name,
                        'long_name': long_name,
                        'definition': definition
                    }

                persistent_diagnoses_ccs_descriptions = dict()
                for icd9_code in set(patient_history[i-1]['new_diagnoses']).union(set(patient_history[i-1]['persistent_diagnoses'])):
                    ccs_code = ccs_dx_mapper.get_ccs_category_for_icd9_code(icd9_code)
                    if ccs_code not in persistent_diagnoses_ccs:
                        continue
                    short_name, long_name, definition = icd_description_cache.get_umls_description_for_mimic_icd9_diagnosis_label(
                        icd9_code)
                    definition = ['' if definition is None else definition]
                    persistent_diagnoses_ccs_descriptions[ccs_code] = {
                        'short_name': short_name,
                        'long_name': long_name,
                        'definition': definition
                    }

            assert len(new_diagnoses_ccs) == len(new_diagnoses_ccs_descriptions)
            assert len(lost_diagnoses_ccs) == len(lost_diagnoses_ccs_descriptions)
            assert len(persistent_diagnoses_ccs) == len(persistent_diagnoses_ccs_descriptions)

            patient_history.append(
                {
                    'hadm_id': current_adm.adm_id,
                    'date': {
                        'weekday': current_adm.adm_date.strftime('%A'),
                        'time': current_adm.adm_date.strftime('%H:%m'),
                        'season': get_season_from_datetime(current_adm.adm_date)
                    },
                    'days_since_first_admission': days_since_first_admission,
                    'new_diagnoses': list(new_diagnoses),
                    'lost_diagnoses': list(lost_diagnoses),
                    'persistent_diagnoses': list(persistent_diagnoses),
                    'new_diagnoses_umls': concepts_to_dicts(new_diagnoses_umls, umls),
                    'lost_diagnoses_umls': concepts_to_dicts(lost_diagnoses_umls, umls),
                    'persistent_diagnoses_umls': concepts_to_dicts(persistent_diagnoses_umls, umls),
                    'new_diagnoses_short_code': list(new_diagnoses_short_code),
                    'lost_diagnoses_short_code': list(lost_diagnoses_short_code),
                    'persistent_diagnoses_short_code': list(persistent_diagnoses_short_code),
                    'new_diagnoses_ccs': list(new_diagnoses_ccs),
                    'new_diagnoses_ccs_descriptions': new_diagnoses_ccs_descriptions,
                    'lost_diagnoses_ccs': list(lost_diagnoses_ccs),
                    'lost_diagnoses_ccs_descriptions': lost_diagnoses_ccs_descriptions,
                    'persistent_diagnoses_ccs': list(persistent_diagnoses_ccs),
                    'persistent_diagnoses_ccs_descriptions': persistent_diagnoses_ccs_descriptions,
                    'new_diagnoses_short_code_umls': new_diagnoses_short_code_umls,
                    'lost_diagnoses_short_code_umls': lost_diagnoses_short_code_umls,
                    'persistent_diagnoses_short_code_umls': persistent_diagnoses_short_code_umls,

                }
            )

        fulltime_persisting_diagnoses = set.intersection(*[set(adm.diagnoses) for adm in admissions])
        fulltime_persisting_diagnoses_umls = set.intersection(*[set(adm.diagnoses_umls) for adm in admissions])
        fulltime_persisting_diagnoses_short_code = set.intersection(*[set([to_3_digit_code(x) for x in adm.diagnoses]) for adm in admissions])
        fulltime_persisting_diagnoses_ccs = set.intersection(*[set([ccs_dx_mapper.get_ccs_category_for_icd9_code(x) for x in adm.diagnoses]) for adm in admissions])

        fulltime_persisting_diagnoses_short_code_umls = []
        all_umls_concepts = set()
        for admission in admissions:
           for concept in admission.diagnoses_umls:
               all_umls_concepts.add(concept)
        all_umls_concepts = collate_umls_set_to_3_digit_code_dict(all_umls_concepts, umls=umls)
        fulltime_persisting_diagnoses_short_code_umls = dict([(x, y) for x, y in all_umls_concepts.items() if x in fulltime_persisting_diagnoses_short_code])


        result.append(
            {
                'subject_id': subject_id,
                'fulltime_persisting_diagnoses': list(fulltime_persisting_diagnoses),
                'fulltime_persisting_diagnoses_umls': concepts_to_dicts(fulltime_persisting_diagnoses_umls, umls),
                'fulltime_persisting_diagnoses_short_code': list(fulltime_persisting_diagnoses_short_code),
                'fulltime_persisting_diagnoses_ccs': list(fulltime_persisting_diagnoses_ccs),
                'fulltime_persisting_diagnoses_short_code_umls': fulltime_persisting_diagnoses_short_code_umls,
                'patient_history': patient_history
            }
        )

    with open(config.path_to_out_readmission_dataset, 'w') as f:
        json.dump(result, f)
