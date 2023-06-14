import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')  
args = parser.parse_args()

num_txt={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5}

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

q_bi=[ 'Is it known whether this drug is administered parenterally?',
 'Is it known whether this drug is applied topically?',
  'Is this compound a small molecule polymer, such as polystyrene sulfonate?',
 'Is this molecule characterized by a small molecular structure or a protein sequence?',
 'Does this compound satisfy the rule-of-three criteria?',
 'Determine if this molecule is inorganic, meaning it contains only metal atoms and fewer than two carbon atoms.',
 'Is there a black box warning associated with this drug?',
 'Is this drug used for therapeutic purposes, rather than for imaging, additives, or other non-therapeutic applications?',
 'Has this approved drug been withdrawn due to toxicity reasons for all indications, populations, and doses in at least one country (not necessarily the US)?',
 'Is it known if this drug is the first approved in its class, regardless of the indication or route of administration, acting on a specific target?',
 'Is it known whether this drug is taken orally?',
 'Is the drug administered in this specific form, such as a particular salt?',
 'Determine if this compound is a prodrug.',
 ]
q_clf=['What is the highest development stage achieved for this compound across all indications? Please respond with Approved, Phase 3 Clinical Trials, Phase 2 Clinical Trials, Phase 1 Clinical Trials, Early Phase 1 Clinical Trials, or Clinical Phase Unknown.',
 'Determine if this drug is administered as a racemic mixture, a single stereoisomer, an achiral molecule, or has an unknown chirality.',
 'Determine the type of availability for this drug.',
 'Is this compound an acid, a base, or neutral?',
 'What is the classification of this molecule? Please respond with Small Molecule, Protein, Antibody, Oligosaccharide, Oligonucleotide, Cell, Enzyme, Gene, or Unknown.',
 ]
q_num=['What is the polar surface area (PSA) value of this compound?',
 "How many violations of Lipinski's Rule of Five are there for this compound, using the HBA_LIPINSKI and HBD_LIPINSKI counts?",
 'What is the calculated ALogP value for this compound?',
 'How many heavy (non-hydrogen) atoms does this compound have?',
 'How many rotatable bonds does this compound have?',
 'How many aromatic rings does this compound have?',
 "How many hydrogen bond acceptors are there in this compound, calculated according to Lipinski's original rules (i.e., counting N and O atoms)?",
 "How many violations of Lipinski's Rule of Five (using HBA and HBD definitions) are there for this compound?",
 'How many hydrogen bond acceptors does this compound have?',
 "How many hydrogen bond donors are there in this compound, calculated according to Lipinski's original rules (i.e., counting NH and OH groups)?",
 'How many hydrogen bond donors does this compound have?',
 'What is the molecular weight of this compound\'s parent molecule?',
 ]
q_sen=['What is the first recorded year of approval for this drug?',
 "What is the definition of this compound's USAN stem?",
 'Which USAN substem can this drug or clinical candidate name be matched with?',
 "Please provide a description of this drug's mechanism of action.",
 'What is the molecular formula of this compound, including any salt that it may have?',]

q_all=list(set(q_bi+q_clf+q_num+q_sen))


#results_file='../chembl_bsl/ChEMBL_QA_test_inference_ep10_new.json'

with open(args.filename) as f:
    chembl=json.load(f)

q_lst={}
q_cnt=[]

for ans_lst in chembl.values():

    for ans in ans_lst:
        if ans[0] in q_lst:
            q_lst[ans[0]].append((ans[1],ans[2]))
            #q_lst[ans[0]].append((ans[1],ans[2][0].split(':')[-1]))
        else:
            q_lst[ans[0]]=[(ans[1],ans[2])]
            #q_lst[ans[0]]=[(ans[1],ans[2][0].split(':')[-1])]
'''
for ans in chembl:
    for q in q_all:
        if q in ans[0]:
            if q in q_lst:
                q_lst[q].append((ans[1],ans[2]))
                #q_lst[ans[0]].append((ans[1],ans[2][0].split(':')[-1]))
            else:
                q_lst[q]=[(ans[1],ans[2])]
'''
for a_lst in q_lst.values():
    q_cnt.append(len(a_lst))

print('Number of questions:{}'.format(str(len(q_cnt))))
print('Number of answers for each question:')
print(q_cnt)

q_res_bi={}

for q in q_bi:
    num_t,num_f=0,0
    for yt,yp in q_lst[q]:
        if yt in yp:num_t+=1
        else:num_f+=1
    q_res_bi[q]=(num_t/(num_t+num_f),num_t,num_f)

q_res_clf={}

for q in q_clf:
    num_t,num_f=0,0
    for yt,yp in q_lst[q]:
        if yt in yp:num_t+=1
        else:num_f+=1
    q_res_clf[q]=(num_t/(num_t+num_f),num_t,num_f)

q_res_num={}

for q in q_num:
    sum_sq,cnt=0,0
    for yt,yp in q_lst[q]:
        yp_lst=yp.split()
        for y in yp_lst:
            #print(y)
            if is_number(y):
                #print(y)
                cnt+=1
                sum_sq+=(yt-float(y))**2
                break
            elif y.lower() in num_txt.keys():
                cnt+=1
                sum_sq+=(yt-num_txt[y.lower()])**2
                break
    if cnt==0:print(q_lst[q])
    q_res_num[q]=(sum_sq/cnt,cnt)

with open('results.txt','w') as f:
    f.write('Binary Questions\n')
    for k,v in q_res_bi.items():
        f.write(k+'\n')
        f.write(str(v)+'\n')
    f.write('\n')

    f.write('Multi-class\n')
    for k,v in q_res_clf.items():
        f.write(k+'\n')
        f.write(str(v)+'\n')
    f.write('\n')

    f.write('Regression\n')
    for k,v in q_res_num.items():
        f.write(k+'\n')
        f.write(str(v)+'\n')