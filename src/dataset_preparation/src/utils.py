import json
import os
from collections import defaultdict
from typing import Dict, List, Union, Optional, Tuple

import psycopg2
from tqdm import tqdm

from src.UMLSParser.UMLSParser import UMLSParser
from src.UMLSParser.model import Concept


class MIMIC2UMLSMatcher:

    def __init__(self, umls: UMLSParser):
        self._PROCEDURE_TUIS = ('T061', 'T058', 'T059', 'T060', 'T169')
        self._umls_concepts_by_icd9_id: Dict[str, List[Concept]] = defaultdict(list)
        for cui, concept in tqdm(umls.get_concepts().items(), desc="Parsing UMLS for ICD9 entities"):
            if 'ICD9CM' in concept.get_source_ids().keys():
                icd9ids = concept.get_source_ids().get('ICD9CM')
                for icd9id in icd9ids:
                    self._umls_concepts_by_icd9_id[icd9id.replace('.', '')].append(concept)

    def get_concept_for_diagnosis(self, identifier: str) -> Union[Concept, None]:
        """
        Returns one concept for the given identifier from the diagnosis table
        :param identifier: MIMIC-style ICD9-CM identifier
        :return: Concept
        """
        if identifier is None:
            return None

        candidates = self._umls_concepts_by_icd9_id[identifier]
        if len(candidates) == 1:
            return candidates[0]
        else:
            candidates = [c for c in candidates if c.get_tui() not in self._PROCEDURE_TUIS]
            if len(candidates) == 1:
                return candidates[0]
        if len(candidates) > 1:
            raise Exception(
                f'Ambiguous match for identifier {identifier}: {", ".join([c.get_tui() for c in candidates])}')
        if len(candidates) == 0:
            return None

    def get_concept_for_procedure(self, identifier: str) -> Union[Concept, None]:
        """
        Returns one concept for the given identifier from the procedures table
        :param identifier: MIMIC-style ICD9-CM identifier
        :return: Concept
        """
        if identifier is None:
            return None

        candidates = self._umls_concepts_by_icd9_id[identifier]
        if len(candidates) == 1:
            return candidates[0]
        else:
            candidates = [c for c in candidates if c.get_tui() in self._PROCEDURE_TUIS]
            if len(candidates) == 1:
                return candidates[0]
        if len(candidates) > 1:
            raise Exception(
                f'Ambiguous match for identifier {identifier}: {", ".join([c.get_tui() for c in candidates])}')
        if len(candidates) == 0:
            return None


class CCSDXMapper:

    def __init__(self, path_to_ccs_dx_folder: str):

        self.__icd9code_2_ccs_category = dict()
        self.__ccs_category_descriptions = dict()

        path_to_ccs_dx_file = os.path.join(path_to_ccs_dx_folder, '$dxref 2015.csv')
        path_to_ccs_dx_label_file = os.path.join(path_to_ccs_dx_folder, 'dxlabel 2015.csv')

        with open(path_to_ccs_dx_label_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 4:
                    continue
                line = line.strip()
                cut = line.find(',')
                ccs_category_id = int(line[:cut])
                description = line[cut + 1:]
                self.__ccs_category_descriptions[ccs_category_id] = description

        with open(path_to_ccs_dx_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:
                    continue
                icd9_code, ccs_category, ccs_category_description, _, _, _ = line.split(',')
                icd9_code = icd9_code[1:-1].strip()
                ccs_category = int(ccs_category[1:-1].strip())
                self.__icd9code_2_ccs_category[icd9_code] = ccs_category

    def get_ccs_category_for_icd9_code(self, icd9_code: str) -> int:
        return self.__icd9code_2_ccs_category[icd9_code]

    def get_description_for_css_category(self, css_category_id: int, key_error: str = 'UNKNOWN',
                                         value_error: str = 'INVALID') -> str:
        try:
            return self.__ccs_category_descriptions[int(css_category_id)]
        except KeyError:
            return key_error
        except ValueError:
            return value_error


class WikidataDescriptionCache:
    def __init__(self, path_to_karls_wikidata_extract: str, ignore_ambigous_matches: bool = True):
        self.__crippled_mimic3_code_definitions = defaultdict(list)
        self.ignore_ambigous_matches: bool = True
        definition_blacklist = ['human disease', 'medical condition', 'wikimedia template', 'disease']

        with open(path_to_karls_wikidata_extract, 'r') as f:
            content = json.load(f)
            for key, icd9_code in content['ICD-9-CM'].items():
                if icd9_code is None:
                    continue
                definition = content['description'][key]
                if definition is None or definition.lower() in definition_blacklist:
                    continue
                self.__crippled_mimic3_code_definitions[icd9_code.replace('.', '')].append(definition)

    def get_definition_for_mimic3_diagnosis_label(self, label: str) -> Optional[list]:
        """
        HINT: This again will suffer from the ambiguous matching because of those crippled MIMIC3 codes.
        :param label:
        :return:
        """
        try:
            if self.ignore_ambigous_matches and len(self.__crippled_mimic3_code_definitions[label]) > 1:
                return None
            else:
                return self.__crippled_mimic3_code_definitions[label]
        except KeyError:
            return None


class ICDDescriptionCache:

    def __init__(self, umls: UMLSParser, cur: psycopg2.extras.DictCursor):
        self.umls_matcher = MIMIC2UMLSMatcher(umls)
        self.diagnoses_source_order = ['NCI', 'MSH', 'HPO', 'NCI_NICHD', 'CSP', 'MEDLINEPLUS', 'NCI_CTCAE',
                                       'NCI_NCI-GLOSS', 'NCI_FDA', 'CHV', 'PSY', 'SNOMEDCT_US', 'NCI_CDISC', 'PDQ',
                                       'CCC', 'NANDA-I', 'AIR', 'JABL', 'HL7V3.0', 'ICF', 'ICF-CY', 'GO', 'UWDA', 'LNC',
                                       'AOT']
        self.icd9_diagnoses_names = dict()

        cur.execute('select icd9_code, short_title, long_title from mimiciii.d_icd_diagnoses')
        for row in cur.fetchall():
            self.icd9_diagnoses_names[row['icd9_code']] = {
                'short_title': row['short_title'],
                'long_title': row['long_title']
            }

    def get_umls_description_for_mimic_icd9_diagnosis_label(self, label: str) -> Tuple[
        Optional[str], Optional[str], Optional[str]]:
        """
        :param label: MIMIC style diagnosis label
        :return: icd9 short_title from MIMIC (if present), icd9 long_title from MIMIC (if present), definition from UMLS (if present)
        """
        try:
            short_title = self.icd9_diagnoses_names[label]['short_title']
            long_title = self.icd9_diagnoses_names[label]['long_title']
        except KeyError:
            short_title, long_title = None, None
        concept: Concept = self.umls_matcher.get_concept_for_diagnosis(label)
        if concept is None:
            return short_title, long_title, None
        definitions = concept.get_definitions()
        if len(definitions) == 0:
            return short_title, long_title, None
        for source_id in self.diagnoses_source_order:
            for definition, source in definitions:
                if source == source_id:
                    return short_title, long_title, definition
        return short_title, long_title, None