import time
import asyncio
from indy import anoncreds, crypto, did, ledger, pool, wallet

import pandas as pd
import numpy as np
import json
import logging
from typing import Optional
import timeit

from indy.error import ErrorCode, IndyError
import shutil
from shutil import rmtree

import os
import subprocess
import signal
from subprocess import call

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("iter", help = 'running iteration')
args = parser.parse_args()

iter = str(args.iter)
#from src.utils import get_pool_genesis_txn_path, run_coroutine

df = pd.DataFrame.from_csv("test_users_dataset.csv", sep='\t')
dforg = pd.DataFrame.from_csv("test_organisation_dataset.csv", sep='\t')

#shutil.rmtree('/home/indy/.indy_client/pool/')
#shutil.rmtree('/home/indy/.indy_client/wallet/')

async def run(iter):

    print("Getting started -> started")

    # Set protocol version 2 to work with Indy Node 1.4
    await pool.set_protocol_version(2)

    # Pool setup
    print("Open Pool Ledger")
    pool_name = 'pool_test'
    pool_config = json.dumps({"genesis_txn": '/home/amine/indy-sdk/cli/docker_pool_transactions_genesis'})

    await pool.create_pool_ledger_config(pool_name, pool_config)

    pool_handle = await pool.open_pool_ledger(pool_name, None)


    print("==============================")
    print("=== Getting Trust Anchor credentials for Faber, Acme, Thrift and Government  ==")
    print("------------------------------")

    #create generic steward
    print("\"Sovrin Steward\" -> Create wallet")
    steward_wallet_name = 'sovrin_steward_wallet'
    steward_wallet_credentials = json.dumps({"key": "steward_wallet_key"})


    await wallet.create_wallet(pool_name, steward_wallet_name, None, None, steward_wallet_credentials)
    steward_wallet = await wallet.open_wallet(steward_wallet_name, None, steward_wallet_credentials)


    print("\"Sovrin Steward\" -> Create and store in Wallet DID from seed")
    steward_did_info = {'seed': '000000000000000000000000Steward1'}
    print('on commence')
    (steward_did, steward_key) = await did.create_and_store_my_did(steward_wallet, json.dumps(steward_did_info))
    print('on sort du bourbié')

    # Create and onbording generic government from dataset
    print("==============================")
    print("== Getting Trust Anchor credentials - Government Onboarding  ==")
    print("------------------------------")

    gen_gov_name = dforg.ix[0, 'Name']
    government_wallet_credentials = json.dumps({"key": "government_wallet_key"})
    government_wallet, steward_government_key, government_steward_did, government_steward_key, _ \
        = await onboarding(pool_handle, pool_name, "Sovrin Steward", steward_wallet,
                           steward_did, gen_gov_name, None, gen_gov_name+"_wallet", government_wallet_credentials)


    print("==============================")
    print("== Getting Trust Anchor credentials - Government getting Verinym  ==")
    print("------------------------------")
    time.sleep(1)
    government_did = await get_verinym(pool_handle, "Sovrin Steward", steward_wallet, steward_did,
                                       steward_government_key, gen_gov_name, government_wallet, government_steward_did,
                                       government_steward_key, 'TRUST_ANCHOR') #set at Trust Anchor

    ###On bording Universities by steward
    #Creating dictionnary containing wallet/did info for all univ

    univ_did_wallet_map = {}
    for i in range(len(dforg)):
        #get Uni name from user dataset
        if dforg.ix[i, 'Type'] == 'University' :
            curr_univ_name = str(dforg.ix[i, 'Name']).replace(" ", "_")
            curr_univ_wallet_name = curr_univ_name+"_wallet"
            curr_univ_wallet_credentials = json.dumps({"key": curr_univ_name+"_wallet_key"})


            ("==============================")
            print("== Getting Trust Anchor credentials - "+curr_univ_name+" Onboarding  ==")
            print("------------------------------")

            curr_univ_wallet, steward_curr_univ_key, curr_univ_steward_did, curr_univ_steward_key, _ = \
            await onboarding(pool_handle,
                             pool_name,
                             "Sovrin Steward",
                             steward_wallet,
                             steward_did,
                             curr_univ_name,
                             None,
                             curr_univ_name+'_wallet',
                             curr_univ_wallet_credentials
                            )


            print("==============================")
            print("== Getting Trust Anchor credentials - " +curr_univ_name+ " getting Verinym  ==")
            print("------------------------------")
            time.sleep(1)
            curr_univ_did = await get_verinym(pool_handle,
                                              "Sovrin Steward",
                                              steward_wallet,
                                              steward_did,
                                              steward_curr_univ_key,
                                              curr_univ_name,
                                              curr_univ_wallet,
                                              curr_univ_steward_did,
                                              curr_univ_steward_key,
                                              'TRUST_ANCHOR'
                                              )

            #Indix list :
            #0 => curr_univ_wallet;
            #1 => curr_univ_wallet_name;
            #2 => steward_curr_univ_key;
            #3 => curr_univ_steward_did;
            #4 => curr_univ_steward_key;
            #5 => curr_univ_did;
            curr_univ_info_list = [curr_univ_wallet, curr_univ_wallet_name, steward_curr_univ_key, curr_univ_steward_did, curr_univ_steward_key, curr_univ_did, curr_univ_wallet_credentials]
            univ_did_wallet_map[curr_univ_name] = curr_univ_info_list

    print(univ_did_wallet_map)
    ###On bording Company by steward
    #Creating dictionnary containing wallet/did info for all comp
    comp_did_wallet_map = {}
    for i in range(len(dforg)):
        #get Uni name from user dataset
        if dforg.ix[i, 'Type'] == 'Company' :
            curr_comp_name = str(dforg.ix[i, 'Name'])
            curr_comp_wallet_name = curr_comp_name+"_wallet"
            curr_comp_wallet_credentials = json.dumps({"key": curr_comp_name+"_wallet_key"})


            print("==============================")
            print("== Getting Trust Anchor credentials - "+curr_comp_name+" Onboarding  ==")
            print("------------------------------")
            time.sleep(1)
            curr_comp_wallet,  steward_curr_comp_key, curr_comp_steward_did, curr_comp_steward_key, _ = \
                await onboarding(pool_handle,
                                 pool_name,
                                 "Sovrin Steward",
                                 steward_wallet,
                                 steward_did,
                                 curr_comp_name,
                                 None,
                                 curr_comp_name+'_wallet',
                                 curr_comp_wallet_credentials
                                )

            print("==============================")
            print("== Getting Trust Anchor credentials - "+curr_comp_name+" getting Verinym  ==")
            print("------------------------------")
            time.sleep(1)
            curr_comp_did = await get_verinym(pool_handle,
                                              "Sovrin Steward",
                                              steward_wallet,
                                              steward_did,
                                              steward_curr_comp_key,
                                              curr_comp_name,
                                              curr_comp_wallet,
                                              curr_comp_steward_did,
                                              curr_comp_steward_key,
                                              'TRUST_ANCHOR'
                                             )
            #Indix list :
            #0 => curr_comp_wallet;
            #1 => curr_comp_wallet_name;
            #2 => steward_curr_comp_key;
            #3 => curr_comp_steward_did;
            #4 => curr_comp_steward_key;
            #5 => curr_comp_did;
            curr_comp_info_list = [curr_comp_wallet, curr_comp_wallet_name, steward_curr_comp_key, curr_comp_steward_did, curr_comp_steward_key, curr_comp_did]
            comp_did_wallet_map[curr_comp_name] = curr_comp_info_list

    ###On bording Bank by steward
    #Creating dictionnary containing wallet/did info for all Bank
    bank_did_wallet_map = {}
    for i in range(len(dforg)):
        #get Uni name from user dataset
        if dforg.ix[i, 'Type'] == 'Bank' :
            curr_bank_name = str(dforg.ix[i, 'Name'])
            curr_bank_wallet_name = curr_bank_name+"_wallet"
            curr_bank_wallet_credentials = json.dumps({"key": curr_bank_name+"_wallet_key"})

            print("==============================")
            print("== Getting Trust Anchor credentials - "+curr_bank_name+" Onboarding  ==")
            print("------------------------------")

            curr_bank_wallet, steward_curr_bank_key, curr_bank_steward_did, curr_bank_steward_key, _ = \
                await onboarding(pool_handle,
                                 pool_name,
                                 "Sovrin Steward",
                                 steward_wallet,
                                 steward_did,
                                 curr_bank_name,
                                 None,
                                 curr_bank_name+'_wallet',
                                 curr_bank_wallet_credentials)

            print("==============================")
            print("== Getting Trust Anchor credentials -"+curr_bank_name+"getting Verinym  ==")
            print("------------------------------")
            time.sleep(1)
            curr_bank_did = await get_verinym(pool_handle,
                                              "Sovrin Steward",
                                              steward_wallet,
                                              steward_did,
                                              steward_curr_bank_key,
                                              curr_bank_name,
                                              curr_bank_wallet,
                                              curr_bank_steward_did,
                                              curr_bank_steward_key,
                                              'TRUST_ANCHOR')
            #Indix list :
            #0 => curr_bank_wallet;
            #1 => curr_bank_wallet_name;
            #2 => steward_curr_bank_key;
            #3 => curr_bank_steward_did;
            #4 => curr_bank_steward_key;
            #5 => curr_bank_did;
            curr_bank_info_list = [curr_bank_wallet, curr_bank_wallet_name, steward_curr_bank_key, curr_bank_steward_did, curr_bank_steward_key, curr_bank_did]
            bank_did_wallet_map[curr_bank_name] = curr_bank_info_list


    ##--------------------------------------------------------------------------
    ###                    CREATE SOME CREDENTIALS SCHEMA                    ###
    ##--------------------------------------------------------------------------
    ###  ==> Government to Ledger

    ## CREDENTIAL SCHEMA #1 : Job Certificate
    print("\"Generic Government\" -> Create \"Job-Certificate\" Schema")
    (job_certificate_schema_id, job_certificate_schema) = \
        await anoncreds.issuer_create_schema(government_did, 'Job-Certificate', '0.2',
                                             json.dumps(['first_name', 'last_name', 'salary', 'employee_status',
                                                         'experience']))

    print("\"Generic Government\" -> Send \"Job-Certificate\" Schema to Ledger")
    await send_schema(pool_handle, government_wallet, government_did, job_certificate_schema)



    ## CREDENTIAL SCHEMA #2 : Transcript
    print("\"Generic Government\" -> Create \"Transcript\" Schema")
    (transcript_schema_id, transcript_schema) = \
        await anoncreds.issuer_create_schema(government_did, 'Transcript', '1.2',
                                             json.dumps(['first_name', 'last_name', 'degree', 'status',
                                                         'year', 'average', 'ssn']))
    print("\"Generic Government\" -> Send \"Transcript\" Schema to Ledger")
    await send_schema(pool_handle, government_wallet, government_did, transcript_schema)


    ## CREDENTIAL SCHEMA #3 : NaN

    ## CREDENTIAL SCHEMA #4 : NaN

    ## CREDENTIAL SCHEMA #5 : NaN

    ## CREDENTIAL SCHEMA #6 : NaN

    ##--------------------------------------------------------------------------
    ###             ORGANIZATIONS SETUP CREDENTIALS DEFINITION               ###
    ##--------------------------------------------------------------------------

    ## -Universities setup their transcipt Credential Definition ==> Schema 2
    univ_cred_map = {} # contains credential def json & id for all uni

    for i in range(len(dforg)):
        #get Uni name from user dataset
        if dforg.ix[i, 'Type'] == 'University' :
            curr_univ_name = str(dforg.ix[i, 'Name']).replace(" ", "_")

            print("==============================")
            print("==="+curr_univ_name+"Credential Definition Setup ==")
            print("------------------------------")

            univ_attr_list = univ_did_wallet_map[curr_univ_name]

            attr_univ_wallet = univ_attr_list[0]
            attr_univ_did = univ_attr_list[5]

            print( curr_univ_name+" -> Get \"Transcript\" Schema from Ledger")
            (_, transcript_schema) = await get_schema(pool_handle, attr_univ_did, transcript_schema_id)

            print(curr_univ_name+" -> Create and store in Wallet \" Transcript\" Credential Definition")
            (curr_univ_transcript_cred_def_id, curr_univ_transcript_cred_def_json) = \
                await anoncreds.issuer_create_and_store_credential_def(attr_univ_wallet, attr_univ_did, transcript_schema,
                                                                        curr_univ_name, 'CL', '{"support_revocation": false}')

            print(curr_univ_name+" -> Send  \" Transcript\" Credential Definition to Ledger")
            await send_cred_def(pool_handle, attr_univ_wallet, attr_univ_did, curr_univ_transcript_cred_def_json)

            #Indix list :
            #0 => curr_univ_transcript_cred_def_id;
            #1 => curr_univ_transcript_cred_def_json;
            curr_univ_cred_def_list = [curr_univ_transcript_cred_def_id, curr_univ_transcript_cred_def_json]
            univ_cred_map[curr_univ_name] = curr_univ_cred_def_list # filling cred_def info for each Uni

    ## -Companies setup their Job application Credential Definition ==> Schema 1
    comp_cred_map = {} # contains credential def json & id for all uni

    for i in range(len(dforg)):
        #get company name from user dataset
        if dforg.ix[i, 'Type'] == 'Company' :
            curr_comp_name = str(dforg.ix[i, 'Name'])

            print("==============================")
            print("==="+curr_comp_name+"Credential Definition Setup ==")
            print("------------------------------")

            comp_attr_list = comp_did_wallet_map[curr_comp_name]

            attr_comp_wallet = comp_attr_list[0]
            attr_comp_did = comp_attr_list[5]


            print("\"Acme\" ->  Get from Ledger \"Job-Certificate\" Schema")
            (_, job_certificate_schema) = await get_schema(pool_handle, attr_comp_did, job_certificate_schema_id)

            print("\"Acme\" -> Create and store in Wallet \"Job-Certificate\" Credential Definition")
            (curr_comp_job_cred_def_id, curr_comp_job_cred_def_json) = \
                await anoncreds.issuer_create_and_store_credential_def(attr_comp_wallet, attr_comp_did, job_certificate_schema,
                                                                        curr_comp_name, 'CL', '{"support_revocation": false}')

            print("\"Acme\" -> Send \"Job-Certificate\" Credential Definition to Ledger")
            await send_cred_def(pool_handle, attr_comp_wallet, attr_comp_did, curr_comp_job_cred_def_json)

            #Indix list :
            #0 => curr_comp_job_cred_def_id;
            #1 => curr_comp_job_cred_def_json;
            curr_comp_cred_def_list = [curr_comp_job_cred_def_id, curr_comp_job_cred_def_json]
            comp_cred_map[curr_comp_name] = curr_comp_cred_def_list # filling cred_def info for each Comp


    ## -Banks setup their ????? Credential Definition ==> NaN



    ##--------------------------------------------------------------------------
    ###             CREATING USERS WALLETS & ON-LEDGER DID                ###
    ##--------------------------------------------------------------------------
    user_info_dict = {}
    for i in range(len(df)) :
        curr_user_fullname = str(df.ix[i, 'first_name']+"_"+df.ix[i, 'last_name'])

        curr_user_wallet_name = curr_user_fullname+'_wallet'
        curr_user_wallet_credentials = json.dumps({"key": "alice_wallet_key"})

        print("==============================")
        print("== Government Create user wallet & did  - Onboarding ==")
        print("------------------------------")

        curr_user_wallet, government_curr_user_key, curr_user_government_did, curr_user_government_key, _ \
            = await onboarding(pool_handle,
                               pool_name,
                               "Generic Government",
                               government_wallet,
                               government_did,
                               curr_user_fullname,
                               None,
                               curr_user_wallet_name,
                               curr_user_wallet_credentials
                              )
        time.sleep(1)
        curr_user_did = await get_verinym(pool_handle,
                                          "Generic Government",
                                          government_wallet,
                                          government_did,
                                          government_curr_user_key,
                                          curr_user_fullname,
                                          curr_user_wallet,
                                          curr_user_government_did,
                                          curr_user_government_key,
                                          None
                                         )

        user_info_list = [curr_user_wallet, curr_user_wallet_name, government_curr_user_key, curr_user_government_did, curr_user_government_key, curr_user_did, curr_user_wallet_credentials]
        user_info_dict[curr_user_fullname] = user_info_list #getting info for every users

    ##--------------------------------------------------------------------------
    ###             SIMULATING AGENTS CREDENTIAL TRANSACTIONS                ###
    ##--------------------------------------------------------------------------

    ## start sniffing packets
    #cap = pyshark.LiveCapture(interface='eth0', output_file='packets.txt', only_summuries=True)
    #cap.sniff(timeout='1000')
    ## I - implement at least 5 type of credential (exchanges & proofs)

    # Defining Credential TYPE I : University transcipt request (user to his uni)  ## LOOPING FOR ALL USERS IN DATASET
    for i in range(len(df)) :
        curr_user_fullname = str(df.ix[i, 'first_name']+"_"+df.ix[i, 'last_name']) # getting user fullname

        #loading attributes from dataset :
        curr_user_firstname = df.ix[i, 'first_name']
        curr_user_lastname = df.ix[i, 'last_name']
        curr_user_univ_name = str(df.ix[i, 'Academics']).replace(" ", "_") # user's alma matter
        curr_user_univ_degree = df.ix[i, 'Academics_Degree']
        curr_user_univ_status = df.ix[i, 'Academics_DegreeStatus']
        curr_user_univ_average = df.ix[i, 'Academics_GradAverage']

        curr_user_snn = df.ix[i, 'SNN']

        print("====================================")
        print("=== User : "+curr_user_fullname+" ==")
        print("====================================")

        print("==============================")
        print("=== Getting Transcript with"+ curr_user_univ_name +"==")
        print("==============================")
        print("== Getting Transcript with "+ curr_user_univ_name +" - Onboarding ==")
        print("------------------------------")
        time.sleep(1)


        #--------Start sniffing subprocess --------:

        label = 'tsp_'+curr_user_fullname+'_'+iter
        sp = subprocess.Popen("exec python3 sniff.py "+label, shell=True)
        #--------Start sniffing subprocess --------:
        time.sleep(0.5)

        curr_user_wallet, curr_univ_user_key, curr_user_univ_did, curr_user_univ_key, curr_univ_user_connection_response \
            = await onboarding(pool_handle,
                               pool_name, curr_user_univ_name,
                               univ_did_wallet_map[curr_user_univ_name][0], #curr_user_univ_wallet
                               univ_did_wallet_map[curr_user_univ_name][5], #curr_user_univ_did
                               curr_user_fullname,
                               user_info_dict[curr_user_fullname][0], #curr_user_wallet
                               curr_user_fullname+'_wallet',
                               user_info_dict[curr_user_fullname][6] #curr_user_wallet_credentials
                               )
        time.sleep(0.2)
        print("==============================")
        print("== Getting Transcript with "+ curr_user_univ_name +" - Getting Transcript Credential ==")
        print("------------------------------")

        print(curr_user_univ_name +" -> Create \"Transcript\" Credential Offer for "+curr_user_fullname)
        transcript_cred_offer_json = \
            await anoncreds.issuer_create_credential_offer(univ_did_wallet_map[curr_user_univ_name][0],
                                                           univ_cred_map[curr_user_univ_name][0])
                                                           #curr_user_univ_transcript_cred_def_id

        print( curr_user_univ_name +" -> Get key for "+curr_user_fullname+"did")
        time.sleep(0.2)

        curr_user_univ_verkey = await did.key_for_did(pool_handle, user_info_dict[curr_user_fullname][0], #user_wallet
                                                      curr_univ_user_connection_response['did'])


        print(curr_user_univ_name +" -> Authcrypt \"Transcript\" Credential Offer for"+curr_user_fullname)
        time.sleep(0.2)

        authcrypted_transcript_cred_offer = await crypto.auth_crypt(univ_did_wallet_map[curr_user_univ_name][0],
                                                                    curr_univ_user_key, curr_user_univ_verkey,
                                                                    transcript_cred_offer_json.encode('utf-8'))

        print(curr_user_univ_name +" -> Send authcrypted \"Transcript\" Credential Offer to "+curr_user_fullname)
        time.sleep(0.2)
        print(curr_user_fullname+" -> Authdecrypted \"Transcript\" Credential Offer from "+ curr_user_univ_name)

        curr_univ_user_verkey, authdecrypted_transcript_cred_offer_json, authdecrypted_transcript_cred_offer = \
            await auth_decrypt(user_info_dict[curr_user_fullname][0],
                               curr_user_univ_key,
                               authcrypted_transcript_cred_offer)

        print(curr_user_fullname+" -> Create and store "+curr_user_fullname+" Master Secret in Wallet")
        time.sleep(0.2)
        curr_user_master_secret_id = await anoncreds.prover_create_master_secret(user_info_dict[curr_user_fullname][0], None)

        print(curr_user_fullname+" -> Get \"Univ Transcript\" Credential Definition from Ledger")
        time.sleep(0.2)
        (curr_univ_transcript_cred_def_id, curr_univ_transcript_cred_def) = \
            await get_cred_def(pool_handle, curr_user_univ_did, authdecrypted_transcript_cred_offer['cred_def_id'])

        print("\"User\" -> Create \"Transcript\" Credential Request for Univ")
        time.sleep(0.2)
        (transcript_cred_request_json, transcript_cred_request_metadata_json) = \
            await anoncreds.prover_create_credential_req(user_info_dict[curr_user_fullname][0], curr_user_univ_did,
                                                     authdecrypted_transcript_cred_offer_json,
                                                     curr_univ_transcript_cred_def,
                                                     curr_user_master_secret_id)

        print("\"User\" -> Authcrypt \"Transcript\" Credential Request for Univ")
        time.sleep(0.2)
        authcrypted_transcript_cred_request = await crypto.auth_crypt(user_info_dict[curr_user_fullname][0], curr_user_univ_key, curr_univ_user_verkey,
                                                                  transcript_cred_request_json.encode('utf-8'))

        print("\"User\" -> Send authcrypted \"Transcript\" Credential Request to Univ")

        print("\"User\" -> Authdecrypt \"Transcript\" Credential Request from Univ")
        time.sleep(0.2)
        curr_user_univ_verkey, authdecrypted_transcript_cred_request_json, _ = \
            await auth_decrypt(univ_did_wallet_map[curr_user_univ_name][0], curr_univ_user_key, authcrypted_transcript_cred_request)

        print("\"Univ\" -> Create \"Transcript\" Credential for User")
        time.sleep(0.2)
        transcript_cred_values = json.dumps({
            "first_name": {"raw": curr_user_firstname, "encoded": "1139481716457488690172217916278103335"}, # How to change encoded values ??????
            "last_name": {"raw": curr_user_lastname , "encoded": "5321642780241790123587902456789123452"},
            "degree": {"raw": curr_user_univ_degree, "encoded": "12434523576212321"},
            "status": {"raw": curr_user_univ_status, "encoded": "2213454313412354"},
            "ssn": {"raw": curr_user_snn, "encoded": "3124141231422543541"},
            "year": {"raw": "2015", "encoded": "2015"},
            "average": {"raw": curr_user_univ_average, "encoded": "5"}
        })

        transcript_cred_json, _, _ = \
            await anoncreds.issuer_create_credential(univ_did_wallet_map[curr_user_univ_name][0],
                                                     transcript_cred_offer_json,
                                                     authdecrypted_transcript_cred_request_json,
                                                     transcript_cred_values, None, None)

        print("\"Univ\" -> Authcrypt \"Transcript\" Credential for User")
        time.sleep(0.2)
        authcrypted_transcript_cred_json = await crypto.auth_crypt(univ_did_wallet_map[curr_user_univ_name][0],
                                                                   curr_univ_user_key, curr_user_univ_verkey,
                                                                   transcript_cred_json.encode('utf-8'))

        print("\"Univ\" -> Send authcrypted \"Transcript\" Credential to User")

        print("\"User\" -> Authdecrypted \"Transcript\" Credential from Univ")
        _, authdecrypted_transcript_cred_json, _ = \
            await auth_decrypt(user_info_dict[curr_user_fullname][0], curr_user_univ_key, authcrypted_transcript_cred_json)

        print("\"User\" -> Store \"Transcript\" Credential from Univ")
        time.sleep(0.2)
        await anoncreds.prover_store_credential(user_info_dict[curr_user_fullname][0], None, transcript_cred_request_metadata_json,
                                            authdecrypted_transcript_cred_json, curr_univ_transcript_cred_def, None)

        #--------Stop sniffing subprocess --------:
        time.sleep(0.5)
        sp.send_signal(signal.SIGINT)
        #--------Stop sniffing subprocess --------:
        time.sleep(0.5)


    # Defining Credential TYPE II : Job certificate request (user to his current company)
    for i in range(len(df)) :
        curr_user_fullname = str(df.ix[i, 'first_name']+"_"+df.ix[i, 'last_name']) # getting user fullname

        curr_user_firstname = str(df.ix[i, 'first_name'])
        curr_user_lastname = str(df.ix[i, 'last_name'])

        curr_user_comp_name = str(df.ix[i, 'Organisation'])

        curr_user_comp_salary = str(df.ix[i, 'Job_Salary'])
        curr_user_comp_experience = str(df.ix[i, 'Job_Experience'])
        curr_user_comp_status = str(df.ix[i, 'Job_Status'])

        print("====================================")
        print("=== User : "+curr_user_fullname+" ==")
        print("====================================")

        print("==============================")
        print("=== Getting Job Certificate with"+ curr_user_comp_name +"==")
        print("==============================")
        print("== Getting Job Certificate with "+ curr_user_comp_name +" - Onboarding ==")
        print("------------------------------")

        time.sleep(0.5)
        #--------Start sniffing subprocess --------:
        label = 'jc_'+curr_user_fullname+'_'+iter
        sp = subprocess.Popen("exec python3 sniff.py "+label, shell=True)
        #--------Start sniffing subprocess --------:
        time.sleep(0.5)

        curr_user_wallet, curr_comp_user_key, curr_user_comp_did, curr_user_comp_key, curr_comp_user_connection_response \
            = await onboarding(pool_handle,
                               pool_name, curr_user_comp_name,
                               comp_did_wallet_map[curr_user_comp_name][0], #curr_user_comp_wallet
                               comp_did_wallet_map[curr_user_comp_name][5], #curr_user_comp_did
                               curr_user_fullname,
                               user_info_dict[curr_user_fullname][0], #curr_user_wallet
                               curr_user_fullname+'_wallet',
                               user_info_dict[curr_user_fullname][6] #curr_user_wallet_credentials

                               )

        print("==============================")
        print("== Getting Job-Certificate Credential from"+curr_user_comp_name+"==")
        print("------------------------------")

        print(curr_user_comp_name+" -> Create \"Job-Certificate\" Credential Offer for User")

        job_certificate_cred_offer_json = \
            await anoncreds.issuer_create_credential_offer(comp_did_wallet_map[curr_user_comp_name][0], #company Wallet
                                                           comp_cred_map[curr_user_comp_name][0])

        time.sleep(1)
        print(curr_user_comp_name+" -> Get key for Alice did")
        curr_user_comp_verkey = await did.key_for_did(pool_handle,
                                                  comp_did_wallet_map[curr_user_comp_name][0],
                                                  curr_comp_user_connection_response['did'])

        print(curr_user_comp_name+" -> Authcrypt \"Job-Certificate\" Credential Offer for User")
        time.sleep(0.2)
        authcrypted_job_certificate_cred_offer = await crypto.auth_crypt(comp_did_wallet_map[curr_user_comp_name][0],
                                                                         curr_comp_user_key,
                                                                         curr_user_comp_verkey,
                                                                         job_certificate_cred_offer_json.encode('utf-8'))

        print( curr_user_comp_name+" -> Send authcrypted \"Job-Certificate\" Credential Offer to User")
        time.sleep(0.2)

        print("\"Alice\" -> Authdecrypted \"Job-Certificate\" Credential Offer from Acme")
        curr_comp_user_verkey, authdecrypted_job_certificate_cred_offer_json, authdecrypted_job_certificate_cred_offer = \
            await auth_decrypt(user_info_dict[curr_user_fullname][0],
                               curr_user_comp_key,
                               authcrypted_job_certificate_cred_offer)


        print("\"User\" -> Create and store \"User\" Master Secret in Wallet") # ATTENTION : DEJA CREE AVANT, à modifier
        time.sleep(0.2)
        curr_user_master_secret_id = await anoncreds.prover_create_master_secret(user_info_dict[curr_user_fullname][0],
                                                                             None)

        print("\"User\" -> Get \"Company Job-Certificate\" Credential Definition from Ledger")
        time.sleep(0.2)
        (_, curr_comp_job_certificate_cred_def) = \
            await get_cred_def(pool_handle, curr_user_comp_did, authdecrypted_job_certificate_cred_offer['cred_def_id'])

        print("\"User\" -> Create and store in Wallet \"Job-Certificate\" Credential Request for Company")
        time.sleep(0.2)
        (job_certificate_cred_request_json, job_certificate_cred_request_metadata_json) = \
            await anoncreds.prover_create_credential_req(user_info_dict[curr_user_fullname][0],
                                                         curr_user_comp_did,
                                                         authdecrypted_job_certificate_cred_offer_json,
                                                         curr_comp_job_certificate_cred_def, curr_user_master_secret_id)

        print("\"User\" -> Authcrypt \"Job-Certificate\" Credential Request for Company")
        time.sleep(0.2)
        authcrypted_job_certificate_cred_request_json = \
            await crypto.auth_crypt(user_info_dict[curr_user_fullname][0],
                                    curr_user_comp_key,
                                    curr_comp_user_verkey,
                                    job_certificate_cred_request_json.encode('utf-8'))

        print("\"User\" -> Send authcrypted \"Job-Certificate\" Credential Request to Company")
        time.sleep(0.2)

        print("\"Company\" -> Authdecrypt \"Job-Certificate\" Credential Request from User")
        curr_user_comp_verkey, authdecrypted_job_certificate_cred_request_json, _ = \
            await auth_decrypt(comp_did_wallet_map[curr_user_comp_name][0],
                               curr_comp_user_key,
                               authcrypted_job_certificate_cred_request_json)

        print("\"Company\" -> Create \"Job-Certificate\" Credential for User")
        time.sleep(0.2)

        curr_user_job_certificate_cred_values_json = json.dumps({
            "first_name": {"raw": curr_user_firstname, "encoded": "245712572474217942457235975012103335"},
            "last_name": {"raw": curr_user_lastname, "encoded": "312643218496194691632153761283356127"},
            "employee_status": {"raw": curr_user_comp_status, "encoded": "2143135425425143112321314321"},
            "salary": {"raw": curr_user_comp_salary, "encoded": "2400"},
            "experience": {"raw": curr_user_comp_experience, "encoded": "10"}
        })

        job_certificate_cred_json, _, _ = \
            await anoncreds.issuer_create_credential(comp_did_wallet_map[curr_user_comp_name][0],
                                                     job_certificate_cred_offer_json,
                                                     authdecrypted_job_certificate_cred_request_json,
                                                     curr_user_job_certificate_cred_values_json, None, None)

        print("\"Company\" ->  Authcrypt \"Job-Certificate\" Credential for User")
        time.sleep(0.2)
        authcrypted_job_certificate_cred_json = \
            await crypto.auth_crypt(comp_did_wallet_map[curr_user_comp_name][0],
                                    curr_comp_user_key, curr_user_comp_verkey,
                                    job_certificate_cred_json.encode('utf-8'))

        print("\"Company\" ->  Send authcrypted \"Job-Certificate\" Credential to user")

        print("\"User\" -> Authdecrypted \"Job-Certificate\" Credential from Company")
        time.sleep(0.2)
        _, authdecrypted_job_certificate_cred_json, _ = \
            await auth_decrypt(user_info_dict[curr_user_fullname][0],
                               curr_user_comp_key,
                               authcrypted_job_certificate_cred_json)

        print("\"User\" -> Store \"Job-Certificate\" Credential")
        time.sleep(0.2)
        await anoncreds.prover_store_credential(user_info_dict[curr_user_fullname][0],
                                                    None,
                                                    job_certificate_cred_request_metadata_json,
                                                    authdecrypted_job_certificate_cred_json,
                                                    comp_cred_map[curr_user_comp_name][1], None)


        #--------Stop sniffing subprocess --------:
        time.sleep(0.5)
        sp.send_signal(signal.SIGINT)
        #--------Stop sniffing subprocess --------:
        time.sleep(0.5)

    # TO DO Defining Credential TYPE III : Gov ID request (user to generic gov)

    # TO DO Defining Credential TYPE IV : Job applying request (user to a random company from our data) (proof)

    # TO DO Defining Credential TYPE V : Loan request (user to a random bank from our data) (proof)



    ## II - simulation strategy : randomly pick a user; randomly pick a credential (exchange or proof); Execute transaction between user and corresponding agent (bank, company or Uni)

    ## III - find a way to store (in a data structure) public on-ledgers relevent data from credentials exchange
    print('PROCESSING FINISH !')

    print("==============================")

    print("Close and Delete pool")
    await pool.close_pool_ledger(pool_handle)
    await pool.delete_pool_ledger_config(pool_name)

    call("rm -rf /home/amine/.indy_client/pool/", shell=True)
    call("rm -rf /home/amine/.indy_client/wallet/", shell=True)


    print("Getting started -> done")


async def onboarding(pool_handle, pool_name, _from, from_wallet, from_did, to,
                     to_wallet: Optional[str], to_wallet_name: str, to_wallet_credentials: str):
    print("\"{}\" -> Create and store in Wallet \"{} {}\" DID".format(_from, _from, to))
    (from_to_did, from_to_key) = await did.create_and_store_my_did(from_wallet, "{}")

    print("\"{}\" -> Send Nym to Ledger for \"{} {}\" DID".format(_from, _from, to))
    await send_nym(pool_handle, from_wallet, from_did, from_to_did, from_to_key, None)

    print("\"{}\" -> Send connection request to {} with \"{} {}\" DID and nonce".format(_from, to, _from, to))
    connection_request = {
        'did': from_to_did,
        'nonce': 123456789
    }

    if not to_wallet:
        print("\"{}\" -> Create wallet".format(to))
        try:
            await wallet.create_wallet(pool_name, to_wallet_name, None, None, to_wallet_credentials)
        except IndyError as ex:
            if ex.error_code == ErrorCode.PoolLedgerConfigAlreadyExistsError:
                pass
        to_wallet = await wallet.open_wallet(to_wallet_name, None, to_wallet_credentials)

    print("\"{}\" -> Create and store in Wallet \"{} {}\" DID".format(to, to, _from))
    (to_from_did, to_from_key) = await did.create_and_store_my_did(to_wallet, "{}")

    print("\"{}\" -> Get key for did from \"{}\" connection request".format(to, _from))
    from_to_verkey = await did.key_for_did(pool_handle, to_wallet, connection_request['did'])

    print("\"{}\" -> Anoncrypt connection response for \"{}\" with \"{} {}\" DID, verkey and nonce"
                .format(to, _from, to, _from))
    connection_response = json.dumps({
        'did': to_from_did,
        'verkey': to_from_key,
        'nonce': connection_request['nonce']
    })
    anoncrypted_connection_response = await crypto.anon_crypt(from_to_verkey, connection_response.encode('utf-8'))

    print("\"{}\" -> Send anoncrypted connection response to \"{}\"".format(to, _from))

    print("\"{}\" -> Anondecrypt connection response from \"{}\"".format(_from, to))
    decrypted_connection_response = \
        json.loads((await crypto.anon_decrypt(from_wallet, from_to_key,
                                              anoncrypted_connection_response)).decode("utf-8"))

    print("\"{}\" -> Authenticates \"{}\" by comparision of Nonce".format(_from, to))
    assert connection_request['nonce'] == decrypted_connection_response['nonce']

    print("\"{}\" -> Send Nym to Ledger for \"{} {}\" DID".format(_from, to, _from))
    await send_nym(pool_handle, from_wallet, from_did, to_from_did, to_from_key, None)

    return to_wallet, from_to_key, to_from_did, to_from_key, decrypted_connection_response


async def get_verinym(pool_handle, _from, from_wallet, from_did, from_to_key,
                      to, to_wallet, to_from_did, to_from_key, role):
    print("\"{}\" -> Create and store in Wallet \"{}\" new DID".format(to, to))
    (to_did, to_key) = await did.create_and_store_my_did(to_wallet, "{}")

    print("\"{}\" -> Authcrypt \"{} DID info\" for \"{}\"".format(to, to, _from))
    did_info_json = json.dumps({
        'did': to_did,
        'verkey': to_key
    })
    authcrypted_did_info_json = \
        await crypto.auth_crypt(to_wallet, to_from_key, from_to_key, did_info_json.encode('utf-8'))

    print("\"{}\" -> Send authcrypted \"{} DID info\" to {}".format(to, to, _from))

    print("\"{}\" -> Authdecrypted \"{} DID info\" from {}".format(_from, to, to))
    sender_verkey, authdecrypted_did_info_json, authdecrypted_did_info = \
        await auth_decrypt(from_wallet, from_to_key, authcrypted_did_info_json)

    print("\"{}\" -> Authenticate {} by comparision of Verkeys".format(_from, to, ))
    assert sender_verkey == await did.key_for_did(pool_handle, from_wallet, to_from_did)

    print("\"{}\" -> Send Nym to Ledger for \"{} DID\" with {} Role".format(_from, to, role))
    await send_nym(pool_handle, from_wallet, from_did, authdecrypted_did_info['did'],
                   authdecrypted_did_info['verkey'], role)

    return to_did

async def send_nym(pool_handle, wallet_handle, _did, new_did, new_key, role):
    nym_request = await ledger.build_nym_request(_did, new_did, new_key, None, role)
    await ledger.sign_and_submit_request(pool_handle, wallet_handle, _did, nym_request)


async def send_schema(pool_handle, wallet_handle, _did, schema):
    schema_request = await ledger.build_schema_request(_did, schema)
    await ledger.sign_and_submit_request(pool_handle, wallet_handle, _did, schema_request)


async def send_cred_def(pool_handle, wallet_handle, _did, cred_def_json):
    cred_def_request = await ledger.build_cred_def_request(_did, cred_def_json)
    await ledger.sign_and_submit_request(pool_handle, wallet_handle, _did, cred_def_request)


async def get_schema(pool_handle, _did, schema_id):
    get_schema_request = await ledger.build_get_schema_request(_did, schema_id)
    get_schema_response = await ledger.submit_request(pool_handle, get_schema_request)
    return await ledger.parse_get_schema_response(get_schema_response)


async def get_cred_def(pool_handle, _did, schema_id):
    get_cred_def_request = await ledger.build_get_cred_def_request(_did, schema_id)
    get_cred_def_response = await ledger.submit_request(pool_handle, get_cred_def_request)
    return await ledger.parse_get_cred_def_response(get_cred_def_response)


async def prover_get_entities_from_ledger(pool_handle, _did, identifiers, actor):
    schemas = {}
    cred_defs = {}
    rev_states = {}
    for item in identifiers.values():
        print("\"{}\" -> Get Schema from Ledger".format(actor))
        (received_schema_id, received_schema) = await get_schema(pool_handle, _did, item['schema_id'])
        schemas[received_schema_id] = json.loads(received_schema)

        print("\"{}\" -> Get Credential Definition from Ledger".format(actor))
        (received_cred_def_id, received_cred_def) = await get_cred_def(pool_handle, _did, item['cred_def_id'])
        cred_defs[received_cred_def_id] = json.loads(received_cred_def)

        if 'rev_reg_seq_no' in item:
            pass  # TODO Create Revocation States

    return json.dumps(schemas), json.dumps(cred_defs), json.dumps(rev_states)


async def verifier_get_entities_from_ledger(pool_handle, _did, identifiers, actor):
    schemas = {}
    cred_defs = {}
    rev_reg_defs = {}
    rev_regs = {}
    for item in identifiers:
        print("\"{}\" -> Get Schema from Ledger".format(actor))
        (received_schema_id, received_schema) = await get_schema(pool_handle, _did, item['schema_id'])
        schemas[received_schema_id] = json.loads(received_schema)

        print("\"{}\" -> Get Credential Definition from Ledger".format(actor))
        (received_cred_def_id, received_cred_def) = await get_cred_def(pool_handle, _did, item['cred_def_id'])
        cred_defs[received_cred_def_id] = json.loads(received_cred_def)

        if 'rev_reg_seq_no' in item:
            pass  # TODO Get Revocation Definitions and Revocation Registries

    return json.dumps(schemas), json.dumps(cred_defs), json.dumps(rev_reg_defs), json.dumps(rev_regs)


async def auth_decrypt(wallet_handle, key, message):
    from_verkey, decrypted_message_json = await crypto.auth_decrypt(wallet_handle, key, message)
    decrypted_message_json = decrypted_message_json.decode("utf-8")
    decrypted_message = json.loads(decrypted_message_json)
    return from_verkey, decrypted_message_json, decrypted_message


if __name__ == '__main__':


    start = timeit.default_timer()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run(iter))

    # shutil.rmtree('/home/amine/.indy_client/pool')
    # shutil.rmtree('/home/amine/.indy_client/wallet')
    
    time.sleep(1)  # FIXME waiting for libindy thread complete

    stop = timeit.default_timer()
    print('Epoch runtime :')
    print(stop - start)
