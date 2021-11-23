import copy
import itertools
import statistics
from collections import Counter

class MitreObject:
    @classmethod
    def print_available_ranks(cls):
        id = 0
        for r in cls.ranks_t:
            print(f'{id}: {r}')
            id += 1


class Datasource(MitreObject):
    metric_keys = ['no_of_techniques', 'total_no_examples', 'total_no_examples_grp', 'total_no_examples_sw', 'total_no_examples_weighted', 'total_no_tactics', 'max_no_examples_weighted', 'min_no_examples_weighted', 'avg_no_examples_weighted', 'median_no_examples_weighted']
    rank_header = ['NO_T', 'R_TOT', 'R_GRP', 'R_SW', 'R_WEIGH', 'R_TAC', 'MAX_WEIGH', 'MIN_WEIGH', 'AVG_WEIGH', 'MEDIAN_WEIGH']
    rank_keys = metric_keys
    rank_val = rank_keys
    ranks_t = ['rank_'+m for m in metric_keys]
    tactic_dict = {}


    def __init__(self, name):
        self.id = name
        self.name = name
        self.fname = ''
        #self.tactic_t_dict = {} # dict that holds techniques within
        self.techniques = []
        self.techniques_copy = []
        self.techniques_in_detection = []
        self.techniques_missing_in_detection = []
        self.no_techniques = 0
        #self.tactic_dist = None
        #self.perm_of_technique = None
        #self.rank_within_tactic = {}

        self.init_rank_dict()

    # basic copy
    def _mycopy(self):
        ds = Datasource(self.name)

        return ds

    # special copy for techniques in detection
    def _mycopy_withtid(self):
        ds = self._mycopy()
        ds.techniques_in_detection = self.techniques_in_detection
        return ds

    def init_rank_dict(self):
        self.metric = {}
        self.rank = {}
        self.rank_header = {}
        for metric, header in zip(Datasource.metric_keys, Datasource.rank_header):
            self.metric[metric] = 0
            self.rank[metric] = 0
            self.rank_header[metric] = header

    @staticmethod
    def do_ranking(ds_dict):
        for metric in Datasource.metric_keys:
            index = 0
            for ds in sorted(sorted(ds_dict.values(), key=lambda x: x.name), key=lambda x: (x.metric[metric], x.metric['no_of_techniques']), reverse=True): #sort by metric, then by metric no_techniques, then name
                index += 1
                ds.rank[metric] = index

    @staticmethod
    def calc_metrics_detection(ds_dict, t_dict, reversable=False):
        for ds in ds_dict.values():
            if reversable:
                ds.techniques_copy = ds.techniques
            ds.techniques = ds.techniques_in_detection

        Datasource.calc_metrics(ds_dict, t_dict)
        if reversable:
            for ds in ds_dict.values():
                ds.techniques = ds.techniques_copy

    @staticmethod
    def calc_metrics(ds_dict, t_dict, ignore_detected=True):
        for ds in ds_dict:
            ds_dict[ds].metric['total_no_examples'] = sum([t_dict[t.id].metric['no_examples'] for t in ds_dict[ds].techniques if ignore_detected or not t_dict[t.id].detected])
            ds_dict[ds].metric['total_no_examples_sw'] = sum([t_dict[t.id].metric['no_examples_sw'] for t in ds_dict[ds].techniques if ignore_detected or not t_dict[t.id].detected])
            ds_dict[ds].metric['total_no_examples_grp'] = sum([t_dict[t.id].metric['no_examples_grp'] for t in ds_dict[ds].techniques if ignore_detected or not t_dict[t.id].detected])
            ds_dict[ds].metric['total_no_examples_weighted'] = round(sum([t_dict[t.id].metric['no_examples_weighted'] for t in ds_dict[ds].techniques if ignore_detected or not t_dict[t.id].detected]),2)

            ds_dict[ds].metric['_tactic_distribution'] = Counter(itertools.chain.from_iterable([t_dict[t.id].tactics for t in ds_dict[ds].techniques if ignore_detected or not t_dict[t.id].detected]))
            ds_dict[ds].metric['total_no_tactics'] = int(len(ds_dict[ds].metric['_tactic_distribution']))
            ds_dict[ds].metric['no_of_techniques'] = len([t for t in ds_dict[ds].techniques if ignore_detected or not t_dict[t.id].detected])
            if ds_dict[ds].metric['no_of_techniques']:
                ds_dict[ds].metric['max_no_examples_weighted'] = max([t_dict[t.id].metric['no_examples_weighted'] for t in ds_dict[ds].techniques if ignore_detected or not t_dict[t.id].detected])
                ds_dict[ds].metric['min_no_examples_weighted'] = min([t_dict[t.id].metric['no_examples_weighted'] for t in ds_dict[ds].techniques if ignore_detected or not t_dict[t.id].detected])
                ds_dict[ds].metric['avg_no_examples_weighted'] = round(statistics.mean([t_dict[t.id].metric['no_examples_weighted'] for t in ds_dict[ds].techniques if ignore_detected or not t_dict[t.id].detected]), 2)
                ds_dict[ds].metric['median_no_examples_weighted'] = round(statistics.median([t_dict[t.id].metric['no_examples_weighted'] for t in ds_dict[ds].techniques if ignore_detected or not t_dict[t.id].detected]), 2)
            ds_dict[ds].no_techniques = len(ds_dict[ds].techniques)

    def get_rank_by_name(self, name):
        if isinstance(name, int):
            return self.get_rank(name)
        return self.get_rank(self.ranks_t.index(name))

    def get_rank(self, index):
        return self.rank[self.metric_keys[index]]

    def get_metric(self, index):
        return self.metric[self.metric_keys[index]]


# this function requires that datasources dict and it's corresponding objects link
# this is done by init_ds_dict_links_set_tactics
def t_dict_datasource_link(ds_dict, t_dict):
    for ds in ds_dict:
        for t in ds_dict[ds].techniques:
            t_dict[t.id].datasources.append(ds_dict[ds])

# this function requires that techniques dict datasources are linked correctly
def ds_dict_techniques_link(ds_dict, t_dict):
    for t in t_dict.values():
        for ds in t.datasources:
            ds_dict[ds.name].techniques.append(t_dict[t.id])

# links are based on information from db entrys (datasources)
def init_ds_dict_links_set_tactics(ds_dict, t_dict):
    # for techniques and datasources, link technique to datasources object
        for t in t_dict.values():
            for ds in ds_dict.keys():
                if ds in t.technique['datasources']:
                    #debug(f'appending {t.technique["id"]} to {ds}')
                    ds_dict[ds].techniques.append(t)
            t.tactics = t.technique['tactic']

def reset_ds_dict_links(ds_dict):
    for ds in ds_dict.values():
        ds.techniques = []

def reset_t_dict_links(t_dict):
    for t in t_dict.values():
        t.datasources = []

def fresh_links(ds_dict, t_dict):
    reset_ds_dict_links(ds_dict)
    reset_t_dict_links(t_dict)
    init_ds_dict_links_set_tactics(ds_dict, t_dict)
    t_dict_datasource_link(ds_dict, t_dict)

class Technique(MitreObject):
    tactic_dict = {}  # static dict that holds objects of technique per tactic

    metric_keys = ['no_examples', 'no_examples_grp', 'no_examples_sw',
                 'no_examples_weighted', 'no_tactics', 'no_datasources'] #DS_VAL_KEY
    rank_header = ['R_TOT', 'R_GRP', 'R_SW',
                 'R_WEIGH', 'R_TAC', 'R_DS']

    rank_keys = metric_keys
    rank_val = rank_keys
    ranks_t = ['rank_'+m for m in metric_keys]

    ds_val_key = ['max_ds_val', 'min_ds_rank']



    def __init__(self, technique, grp_weight = 0):
        self.id = technique['id']
        self.name = technique['name']
        self.technique = technique
        self.fname = ''
        self.tactics = []
        self.datasources = []
        self.datasource_val = {}
        self.custom_db_import = False

        self.grp_weight = grp_weight  # save weight of grp to technique
        self.detected = False

        self.metric = {}
        self.rank = {}
        self.rank_header = {}
        self.init_rank_dict()

    # IMPLEMENT COPY PROCEDURE
    def _mycopy(self):
        t = Technique(self.technique, grp_weight=self.grp_weight)
        return t

    def _mycopy_withval(self):
        t = self._mycopy()
        t.tactics = self.tactics
        #t.detected = self.detected
        t.custom_db_import = self.custom_db_import
        t.metric = copy.deepcopy(self.metric)
        t.rank = copy.deepcopy(self.rank)
        return t


    def init_rank_dict(self):
        for (metric, header) in zip(Technique.metric_keys, Technique.rank_header):
            self.metric[metric] = 0
            self.rank[metric] = 0
            self.rank_header[metric] = header


    def get_max_ds_score(self, key=Datasource.rank_keys[4]):
        max = 0
        for ds in self.datasources:
            newmax = ds.metric[key]
            if newmax > max:
                max = newmax

        return max

    def clean_mitre_info(self):
        self.technique['grp_examples'] = []
        self.technique['sw_examples'] = []
        self.technique['no_examples_sw'] = 0
        self.technique['no_examples_grp'] = 0
        self.technique['no_examples'] = 0

    def get_rank_by_name(self, name):
        if isinstance(name, int):
            return self.get_rank(name)
        return self.get_rank(self.ranks_t.index(name))

    def get_rank(self, index):
        return self.rank[self.metric_keys[index]]

    def get_metric(self, index):
        return self.metric[self.metric_keys[index]]

    @staticmethod
    def do_ranking(t_dict):
        for metric in Technique.metric_keys:
            index = 0
            for t in sorted(sorted(t_dict.values(), key=lambda x: x.name), key=lambda x: x.metric[metric], reverse=True):
                index += 1
                t.rank[metric] = index

    # this method removes datasources already in detection
    @staticmethod
    def rebuild_t_dict_from_ds_for_detection(ref_ds_dict, ref_t_dict):
        for t in ref_t_dict.values():
            t.datasources = []

        for ds in ref_ds_dict.values():
            for t in ds.techniques:
                t.datasources.append(ds)

        for t in ref_t_dict.values():
            t.no_datasources = len(t.datasources)

    # calc weighted no examples and rank
    # also get no examples and grp examples to object and rank
    # use this only when using DB values from mitre
    @staticmethod
    def get_n_calc_metric(t_dict):
        for t in t_dict.values():
            t.metric['no_examples_sw'] = int(t.technique['no_examples_sw'])
            b = int(t.technique['no_examples_grp']) * t.grp_weight
            t.metric['no_examples_weighted'] = round(t.metric['no_examples_sw'] + b, 2)  # percentage of total coverage
            t.metric['no_examples'] = int(t.technique['no_examples'])
            t.metric['no_examples_grp'] = int(t.technique['no_examples_grp'])
            t.metric['no_tactics'] = int(len(t.technique['tactic']))
            t.metric['no_datasources'] = int(len(t.technique['datasources']))
            t.no_datasources = len(t.datasources)
            ## NO OF DATA SOURCES

            # THIS IS FOR COMPATBILITY, REMOVE IN FUTURE VERSIONS
            #for metric in Technique.metric_keys:
            #    setattr(t, metric, t.metric[metric])
