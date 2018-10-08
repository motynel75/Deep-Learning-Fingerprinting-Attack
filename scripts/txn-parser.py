import json
import numpy as np


#This script extracts and regroup information from ledger data
def has_attribute(data, attribute):
    return attribute in data and data[attribute] is not None

txs = []
ids = []
nym = []
attrib = []
schema = []
claim_def = []
other = []

nym_features = {}       # key = seqNo | contains txnTime, from did, role, dest did
schema_features = {}    # key = seqNo | contains txnTime, from did, attr_name, name
claim_def_features = {} # key = seqNo | contains txnTime, from did, primary data, tag

with open('reader_ledger-domain.txt', 'r') as raw_file:
    for line in raw_file:
        line = line.rstrip('\n')
        line = line.split(' ', 1)[1]
        tx = json.loads(line)


        if tx["txn"]["type"] == "1":
            nym.append(tx)
            txTime = ''
            txFromDid =''
            txRole =''
            txDest =''

            if has_attribute(tx["txnMetadata"], 'txnTime') :
                txTime = tx["txnMetadata"]["txnTime"]

            if has_attribute(tx["txn"]["metadata"], 'from') :
                txFromDid = tx["txn"]["metadata"]["from"]

            if has_attribute(tx["txn"]["data"], 'role') :
                txRole = tx["txn"]["data"]["role"]

            if has_attribute(tx["txn"]["data"], 'dest') :
                txDest = tx["txn"]["data"]["dest"]

            nym_features[tx["txnMetadata"]["seqNo"]]=[ txTime, txFromDid, txRole, txDest]


        elif tx["txn"]["type"] == "100":
            attrib.append(tx)

        elif tx["txn"]["type"] == "101":
            schema.append(tx)
            txTime = ''
            txFromDid =''
            txAttrName =''
            txName =''

            if has_attribute(tx["txnMetadata"], 'txnTime') :
                txTime = tx["txnMetadata"]["txnTime"]

            if has_attribute(tx["txn"]["metadata"], 'from') :
                txFromDid = tx["txn"]["metadata"]["from"]

            if has_attribute(tx["txn"]["data"], 'attr_names') :
                txAttrName = tx["txn"]["data"]["attr_names"]

            if has_attribute(tx["txn"]["data"], 'name') :
                txName = tx["txn"]["data"]["name"]  #orgs label

            schema_features[tx["txnMetadata"]["seqNo"]]=[ txTime, txFromDid, txAttrName, txName]

        elif tx["txn"]["type"] == "102":
            claim_def.append(tx)
            txTime = ''
            txFromDid =''
            txPrimaryData =''
            txTag =''

            if has_attribute(tx["txnMetadata"], 'txnTime') :
                txTime = tx["txnMetadata"]["txnTime"]

            if has_attribute(tx["txn"]["metadata"], 'from') :
                txFromDid = tx["txn"]["metadata"]["from"]

            if has_attribute(tx["txn"]["data"], 'data') and has_attribute(tx["txn"]["data"]["data"], 'primary') :
                txPrimaryData = tx["txn"]["data"]["data"]["primary"]

            if has_attribute(tx["txn"]["data"], 'tag') :
                txTag = tx["txn"]["data"]['tag']    #user label

            claim_def_features[tx["txnMetadata"]["seqNo"]]=[ txTime, txFromDid, txPrimaryData, txTag]

        if tx["txn"]["metadata"]:
            ids.append(tx["txn"]["metadata"]["from"])
        txs.append(tx)
print("Unique IDs {}".format(len(set(ids))))
print("NYM {}".format(len(nym)))
print("Attributes {}".format(len(attrib)))
print("Schema {}".format(len(schema)))
print("Claim def {}".format(len(claim_def)))
print("Total txs {}".format(len(txs)))


# Analyse the claim definitions
claim_ids = []
for claim in claim_def:
        if claim["txn"]["metadata"]:
            claim_ids.append(claim["txn"]["metadata"]["from"])
print("Claim Ids {}".format(len(set(claim_ids))))

print(len(claim_def)) #unique from id
print(len(claim_ids)) #unique from id
print(len(ids))

unique_claim_ids = np.unique(claim_ids)
unique_ids = np.unique(ids)


from_did_map = {}

for did in unique_ids:
    seq_list = []
    for seq in range(1,len(txs)):

        if seq in claim_def_features.keys() and did in claim_def_features[seq] :
            seq_list.append(claim_def_features[seq])

        if seq in nym_features.keys() and did in nym_features[seq] :
            seq_list.append(nym_features[seq])

        if seq in schema_features.keys() and did in schema_features[seq] :
            seq_list.append(schema_features[seq])

    from_did_map[did] = seq_list

print(from_did_map)
