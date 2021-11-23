from stix2 import FileSystemSource
from stix2 import Filter
from stix2.utils import get_type_from_id
#from attackcti import attack_client
import collections
import yaml
import logging
import concurrent.futures


#fs = FileSystemSource('../stix-data/cti-ATT-CK-v9.0/enterprise-attack/')
logging.getLogger().setLevel(logging.INFO)



def get_group_by_alias(src):
    return remove_revoked_deprecated(src.query([
        Filter('type', '=', 'intrusion-set'),
    ]))

def get_techniques(src):
    return remove_revoked_deprecated(src.query([
        Filter('type', '=', 'attack-pattern')]))

def get_sw(src):
    swgrp = src.query([
        Filter('type', 'in', ['malware', 'tool']),
    ])
    return remove_revoked_deprecated(swgrp)
def get_grp(src):
    swgrp = src.query([
        Filter('type', '=', 'intrusion-set'),
    ])
    return remove_revoked_deprecated(swgrp)

def get_grpsw_by_technique(src, t_id):
    t_rel = src.query([
        Filter('type', '=', 'relationship'),
        Filter('relationship_type', '=', 'uses'),
        Filter('target_ref', '=', t_id)
    ])
    t_rel = remove_revoked_deprecated(t_rel)
    return [r['source_ref'] for r in t_rel]

def get_swgrp_by_id(src, swgrp_id):
    logging.debug(f'getting swgrp {swgrp_id}')
    swgrp = src.query([
        Filter('type', 'in', ['intrusion-set', 'malware', 'tool']),
        Filter('id', '=', swgrp_id)
    ])
    swgrp = remove_revoked_deprecated(swgrp)
    if(len(swgrp) == 0):
        logging.info(f"{swgrp_id} is revoked")
        return None
    return swgrp[0]

def get_maintechnique(src, t_id):
    logging.debug(f'getting maintechnique for {t_id}')
    t_rel = src.query([
        Filter('type', '=', 'relationship'),
        Filter('relationship_type', '=', 'subtechnique-of'),
        Filter('source_ref', '=', t_id)
    ])
    t_rel = remove_revoked_deprecated(t_rel)
    t = src.query([
        Filter('type', '=', 'attack-pattern'),
        Filter('id', '=', t_rel[0].target_ref)
    ])
    t = remove_revoked_deprecated(t)
    if(len(t) == 0):
        logging.info(f"{t} is revoked")
        return None
    return t[0]


def try_add_technique(db, key, where, what):
    try:
        db[key][where]['techniques'].append(what)
    except:
        db[key][where] = {}
        db[key][where]['techniques'] = []
        db[key][where]['techniques'].append(what)

def dict_add_ds(db, t):
    if len(t['datasources']) == 0:
        try_add_technique(db, 'datasource', 'NO_DATASOURCE', t['id']) # empty db name / no db for tech

    for ds in t['datasources']:
        try_add_technique(db, 'datasource', ds, t['id'])

def dict_add_group(db, t):
    for grp in t['grp_examples']:
        try_add_technique(db, 'group', grp, t['id'])

def dict_add_sw(db, t):
    for sw in t['sw_examples']:
        try_add_technique(db, 'software', sw, t['id'])

def dict_add_tactic(db, t):
    for tac in t['tactic']:
        try_add_technique(db, 'tactic', tac, t['id'])


def _build_db(db, fs):

    techniques = get_techniques(fs)
    db['general']['total_techniques'] = len(techniques)

    i = 0
    for t in techniques:
        subt = False
        i += 1
        software_examples = []
        group_examples = []
        tdict = {}
        ds = []

        logging.debug(f'starting with technique {t["id"]}')
        if 'x_mitre_is_subtechnique' in t:
            subt = t['x_mitre_is_subtechnique']

        if 'x_mitre_data_sources' in t:
            ds = t['x_mitre_data_sources']

        t_name = t['name']
        if (subt):
            t_name = f'{get_maintechnique(fs, t["id"])["name"]}: {t_name}'  # build full name

        for g in get_grpsw_by_technique(fs, t['id']):
            g_obj = get_swgrp_by_id(fs, g)
            if not g_obj: #continue if revoked
                continue
            if (g_obj['type'] in ['malware', 'tool']):
                software_examples.append(g_obj['external_references'][0]['external_id'])
            elif (g_obj['type'] in ['intrusion-set']):
                group_examples.append(g_obj['external_references'][0]['external_id'])
            else:
                assert ("something went wrong, this is impossible")

        tdict['id'] = t['external_references'][0]['external_id']
        tdict['is_subtechnique'] = subt
        tdict['name'] = t_name
        tdict['created'] = str(t['created'])
        tdict['last_mod'] = str(t['modified'])
        tdict['no_examples'] = len(software_examples) + len(group_examples)
        tdict['no_examples_grp'] = len(group_examples)
        tdict['grp_examples'] = group_examples
        tdict['no_examples_sw'] = len(software_examples)
        tdict['sw_examples'] = software_examples
        tdict['tactic'] = [p['phase_name'] for p in t['kill_chain_phases']]
        tdict['url'] = t['external_references'][0]['url']
        tdict['datasources'] = ds

        logging.debug(f'almost done with technique {t["id"]}')

        dict_add_ds(db, tdict)
        dict_add_group(db, tdict)
        dict_add_sw(db, tdict)

        logging.debug(f'done with technique {t["id"]}')

        db['technique'][tdict['id']] = tdict
        #if i >= 10:
        #    break


def remove_revoked_deprecated(stix_objects):
    return list(
        filter(
            lambda x: x.get("x_mitre_deprecated", False) is False and x.get("revoked", False) is False,
            stix_objects
        )
    )

def dump_stix_to_yaml(src, target):
    database = {
        'general': {},
        'tactic': {},
        'technique': {},
        'datasource': {},
        'group': {},
        'software': {},
    }
    fs = FileSystemSource(src)
    logging.info(f'building database using {src}')
    _build_db(database, fs)
    logging.info(f'writing database {src} to {target}')
    with open(target, 'w') as file:
        yaml.dump(database, file, default_flow_style=False)
    logging.info(f'done writing db to {target}')

def spin_threads(list):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for item in list:
            logging.info(f'firing up thread {item}')
            executor.submit(dump_stix_to_yaml, item[0], item[1])

#dump_stix_to_yaml('/tmp/stix-data/cti-ATT-CK-v1.0/enterprise-attack/', '../stix-data/cti-ATT-CK-v1.0.yaml')

spin_threads([('/tmp/stix-data/cti-ATT-CK-v9.0/enterprise-attack/', '../stix-data/cti-ATT-CK-v9.0.yaml'),
('/tmp/stix-data/cti-ATT-CK-v8.2/enterprise-attack/', '../stix-data/cti-ATT-CK-v8.2.yaml'),
('/tmp/stix-data/cti-ATT-CK-v8.1/enterprise-attack/', '../stix-data/cti-ATT-CK-v8.1.yaml'),
('/tmp/stix-data/cti-ATT-CK-v8.0/enterprise-attack/', '../stix-data/cti-ATT-CK-v8.0.yaml'),
('/tmp/stix-data/cti-ATT-CK-v7.2/enterprise-attack/', '../stix-data/cti-ATT-CK-v7.2.yaml'),
('/tmp/stix-data/cti-ATT-CK-v7.1/enterprise-attack/', '../stix-data/cti-ATT-CK-v7.1.yaml'),
('/tmp/stix-data/cti-ATT-CK-v7.0/enterprise-attack/', '../stix-data/cti-ATT-CK-v7.0.yaml'),
('/tmp/stix-data/cti-ATT-CK-v6.3/enterprise-attack/', '../stix-data/cti-ATT-CK-v6.3.yaml'),
('/tmp/stix-data/cti-ATT-CK-v6.2/enterprise-attack/', '../stix-data/cti-ATT-CK-v6.2.yaml'),
('/tmp/stix-data/cti-ATT-CK-v6.1/enterprise-attack/', '../stix-data/cti-ATT-CK-v6.1.yaml'),
('/tmp/stix-data/cti-ATT-CK-v6.0/enterprise-attack/', '../stix-data/cti-ATT-CK-v6.0.yaml'),
('/tmp/stix-data/cti-ATT-CK-v5.2/enterprise-attack/', '../stix-data/cti-ATT-CK-v5.2.yaml'),
('/tmp/stix-data/cti-ATT-CK-v5.1/enterprise-attack/', '../stix-data/cti-ATT-CK-v5.1.yaml'),
('/tmp/stix-data/cti-ATT-CK-v5.0/enterprise-attack/', '../stix-data/cti-ATT-CK-v5.0.yaml'),
('/tmp/stix-data/cti-ATT-CK-v4.0/enterprise-attack/', '../stix-data/cti-ATT-CK-v4.0.yaml'),
('/tmp/stix-data/cti-ATT-CK-v3.0/enterprise-attack/', '../stix-data/cti-ATT-CK-v3.0.yaml'),
('/tmp/stix-data/cti-ATT-CK-v2.0/enterprise-attack/', '../stix-data/cti-ATT-CK-v2.0.yaml'),
('/tmp/stix-data/cti-ATT-CK-v1.0/enterprise-attack/', '../stix-data/cti-ATT-CK-v1.0.yaml')])


def debug_fun():
    fs = FileSystemSource('/tmp/stix-data/cti-ATT-CK-v9.0/mobile-attack/')

    techniques = get_techniques(fs)
    techniques = remove_revoked_deprecated(techniques)
    t_num=0
    st_num=1
    for t in techniques:
        if 'x_mitre_is_subtechnique' in t:
            if t['x_mitre_is_subtechnique'] == False:
                #print(t['external_references'][0]['external_id'])
                t_num+=1
            if t['x_mitre_is_subtechnique'] == True:
                st_num+=1
        else:
            t_num+=1

    sw = get_sw(fs)
    grp = get_grp(fs)

    print(t_num)
    print(st_num)
    # this differs from attack site because there is no speration of grp and software per matrix, e.g. some groups are only using tactics of mobile matrix
    print(len(sw))
    print(len(grp))
    print([g['name'] for g in grp])

    #get_swgrp_by_id(fs, 'intrusion-set--dc5e2999-ca1a-47d4-8d12-a6984b138a1b')

#debug_fun()

#_build_db()
#print(yaml.dump(database, default_flow_style=False))
