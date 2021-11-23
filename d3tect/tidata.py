import yaml
import glob

from logging import getLogger, debug, info, warn, error, DEBUG, INFO
from d3tect.helper import _read_yaml_file

class TIData:
    ranked_list = None
    no_dbs = 0 # Number of databases
    dbs = [] # Names of databases

    def __init__(self):
        if self.ranked_list == None: # build list
            self._get_technique_ranking_from_database()

    # returns ranked lists of techniques
    def _get_technique_ranking_from_database(self, t_limit=9999, path='DeTTECT/threat-actor-data/', include_subt=True):
        dbs = sorted(glob.glob(f"{path}/*.yaml")) # get all yaml files in path
        new_list = []

        self.dbs = []
        self.no_dbs = len(dbs)

        for db in dbs:
            db_tmp = _read_yaml_file(db) # read database
            if db_tmp:
                obj = db_tmp['groups'][0]
                techniques = obj['technique_id']
                sorted_techniques = [] # this is a sorted list of techniques including rank [technique, rank, value]
                total_no_techniques = len(techniques)
                rank = 0
                if isinstance(techniques, dict):
                    for id, val in sorted(techniques.items(), key=lambda x: (x[1], x[0]), reverse=True): # sort by val and then by name
                        rank += 1
                        if(rank > t_limit):
                            break
                        sorted_techniques.append([id, rank, val])
                elif isinstance(techniques, list):
                    for x in techniques:
                        sorted_techniques.append([x, 'x', '0'])
                else:
                    raise Exception("This is weird. We don't know this format so we are stopping here.")

                while obj['group_name'] in self.dbs: # make sure group names do not exist twice
                    obj['group_name'] = f'{obj["group_name"]}*'

                new_list.append({
                    'full_name': obj['group_name'],
                    'short': obj['group_name'].split('-')[0].strip(),
                    'no_techniques': total_no_techniques,
                    'sorted_techniques': sorted_techniques,
                })
                self.dbs.append(obj['group_name'])

        if(t_limit == 9999): # only cache complete dumps as ranked list
            self.ranked_list = new_list

        return new_list


    # returns technique rank
    def _get_technique_rank_from_database(self, id):
        result = {}
        for db in self.ranked_list:
            for x in db['sorted_techniques']:
                if x[0] == id:
                    result.update({db['full_name']: x[1]})

        return result

    def _get_missing_dbs_for_technique(self, id):
        databases = []
        for db in self.ranked_list:
            for x in db['sorted_techniques']:
                if x[0] == id:
                    databases.append(db['full_name']) # build list of databases for technique
        result = [x for x in self.dbs if x not in databases]

        return result

    def _get_missing_techniques_for_db(self, lo_techniques, db_index):
        #print(self.ranked_list[db_index]['sorted_techniques'])

        return [x for x in self.ranked_list[db_index]['sorted_techniques'] if x[0] not in lo_techniques]
