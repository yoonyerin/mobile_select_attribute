import pandas as pd
import numpy as np
from config import *
import matplotlib.pyplot as plt 

# service_attribute_names=["hair_color" ,'Receding_Hairline', 
#                   'Narrow_Eyes', 'Pointy_Nose',  'Bushy_Eyebrows','Arched_Eyebrows', 'Big_Nose',  
#                   'Male', 'High_Cheekbones', "Pale_Skin", "Oval_Face", "Bags_Under_Eyes", "Big_Lips"]
# reverse_attribute_names=['Narrow_Eyes', 'Pointy_Nose',  'Bushy_Eyebrows','Arched_Eyebrows', 'Big_Nose']
hair_color=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']

label=pd.read_csv("/Users/yerinyoon/Documents/cubig/mobile_attribute_select/data/list_attr_celeba.csv")
label=label.drop("image_id", axis=1)

#black 1 , blond 2 , gray 3, brown 4

def proportion_chk(attr):
    ann_file=pd.read_csv(attr)
    ann_file=ann_file.iloc[:, 1:]
    ann_file=ann_file.replace(-1, 0)
    proportion=ann_file.sum()*100/ann_file.shape[0]
    
    imbalance={}
    total={}
    for i in range(ann_file.shape[1]):
        total[ann_file.columns[i]]=proportion[i]
        if abs(proportion[i])<20: 
            imbalance[ann_file.columns[i]]=proportion[i]
                   
    print(imbalance)
    imbalance=pd.Series(imbalance)
    imbalance.to_csv("./imbalance_attribute.csv")
    
    total=pd.Series(total)
    total=total.sort_values(ascending=True)
    total.to_csv("./total.csv")
    
    X_axis = np.arange(ann_file.shape[1])
    X=ann_file.columns
    
    plt.bar(X_axis - 0.4, proportion, 0.4, label = 'The number of the group')
    #plt.bar(X_axis - 0.2, y[1], 0.4, label = 'Service', color=service_color)

    plt.xticks(X_axis, X)
    plt.xlabel("Attribute")
    plt.ylabel("Proportion")
    plt.title(f"The proportion of attributes")
    plt.legend()
    plt.show()    

def hair_color_anomaly(attr):
    ann_file=pd.read_csv(attr)
    ann_file=ann_file.iloc[:, 1:]
    onehot=["1000", "0100", "0010", "0001"]
    ann_file=ann_file.replace(-1, 0)
    for i in hair_color:
        ann_file=ann_file.astype({i:"str" })   
    ann_file["hair_color"]=ann_file["Black_Hair"]+ann_file["Blond_Hair"]\
            +ann_file["Gray_Hair"]+ann_file["Brown_Hair"]
    hair_vector=ann_file["hair_color"].value_counts(ascending=False)
        
    X_axis = np.arange(len(hair_vector.values))
    X=[str(i) for  i in hair_vector.keys()]    
    
    colors=[]
    for i in X:
        if i in onehot:
            colors.append("orange")
        else:
            colors.append("red")
    #ann_file.drop(hair_color,axis=1 ,inplace=True)
        
    plt.bar(X_axis - 0.4, hair_vector.values, 0.4, label = 'The number of the group', color=colors)
    #plt.bar(X_axis - 0.2, y[1], 0.4, label = 'Service', color=service_color)

    plt.xticks(X_axis, X)
    plt.xlabel("Hair Color Partition")
    plt.ylabel("The number of the group")
    plt.title(f"Hair Color anonaly??")
    plt.legend()
    plt.show()    
            # vect에 포함되지 않으면?? 
    

        

def corr_ranking(fixed_attr, data, columns=None):
    if columns==None:
        columns=label.columns
        # ann_file=pd.read_csv(ATTRIBUTE_FILE)
        # ann_file=ann_file.iloc[:, 1:]
        # columns=ann_file.columns
    corr_list=[]
    for i in columns:
        corr_data=pd.concat([data[fixed_attr], data[i]], axis=1)
        try:
            corr=corr_data.corr().iloc[1,0]
            corr_list.append(np.abs(corr))
        except: 
            print("Are you sure you are using onehot labels data?")
            print(f"PASS: {i}")

    corr=pd.Series(corr_list, index=columns)

    corr_rank=corr.sort_values(ascending=True)
    print(corr_rank)
    corr_rank.to_csv(f"./{fixed_attr}_corrlation.csv")
    return corr_rank


# rank=corr_ranking("Pale_Skin", label, ["Receding_Hairline"])
# print(rank)
# rank.to_csv("./hair_color_corrlation.csv")
# 


## scenario
### 1. file을 가져옴
### 2. hair clolor vector 그룹을 원핫인코딩 상태에서 넘버링으로 바꿈
### 3. -1 => 0으로 변환하여 포멀한 바이너리 데이터로 만듬
### 4. 
#### 1) exclude: 헤어 커러들 포함하지 않기
#### 2) include: 헤어커러들 포함하기(default)
### 5. 그루핑 정보 시각화
def service_dataset_distribution(attr, mode="include", chosen=[], limit=100, fix=0, hair_priority=[0, 1, 2, 3, 4]):
    
    
    ann_file=pd.read_csv(attr)
    ann_file=ann_file.iloc[:, 1:]
    
    ## len(onehot)=12
    ###onehot: 'Nothing', Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', "Blond", "Black", "Blond"
    onehot=["0000", "1000", "0100", "0010", "0001",  "0101", "1001", "0110", "0011", "1010", "0111","1100"]


    temp=[[0], [1], [2], [3], [4], [2, 4],[1, 4], [2, 3], [3, 4], [1, 3], [2, 3, 4], [1, 2]]
    #numbering=np.zeros([6, 12])
    numbering=[]
    
    #for i in range(len(hair_priority)):
    #print(hair_priority[i])
    
    for cnt, i in enumerate(temp): #vector
        for index in i: #possibility
            for j in hair_priority[fix]: #priority chk
                if j==index:
                    numbering.append(j)
                    continue
            if len(numbering)!=cnt+1:
                numbering.append(0)
        
                    
            
            
            
    
    ann_file=ann_file.replace(-1, 0)
    print(ann_file.shape)
    if mode=="exclued":
        
        
        group_count=ann_file[chosen].value_counts(ascending=False)
        keys=group_count.keys()
        values=group_count.values
        
        names=chosen
        # keys=reverse_attr_key
        # values=reverse_attr_values
        ##expected lower bound of the data
        
        X_axis = np.arange(len(keys))
        X=[str(i) for  i in keys]       
        
    else: 
        # for i in 
        #ann_file["hair_color"]=0
        #for i in hair_color:
        #     if ann_file[i]==1:
        #         ann_file["hair_color"]= [i for i, color in enumerate(hair_color) if ann_file[color]==1]
        ##TODO: hair color를 원핫 => 한 컬럼으로 
        #for i in hair_color:
        
        for i in hair_color:
            ann_file=ann_file.astype({i:"str" })   
        ann_file["hair_color"]=ann_file["Black_Hair"]+ann_file["Blond_Hair"]\
            +ann_file["Gray_Hair"]+ann_file["Brown_Hair"]
        
        for i in range(len(onehot)):
            ann_file["hair_color"]=ann_file["hair_color"].replace(onehot[i], numbering[i])
            # vect에 포함되지 않으면?? 
        
        ann_file.drop(hair_color,axis=1 ,inplace=True)
        
        group_count=ann_file[chosen].value_counts(ascending=False)   
        selected_attr_keys=group_count.keys()
        print(type(chosen[0]))
        selected_attr_values=group_count.values
        print(f"sum: {sum(selected_attr_values)}")
        
         
        keys=selected_attr_keys
        values=selected_attr_values
        names=chosen
        group_len=len(keys)
        
        X_axis = np.arange(len(selected_attr_keys))
        X=[str(i) for  i in selected_attr_keys]
    
    #print(len(keys))
    
    #print(40-len(keys)-len(hair_color)+1)
    expected_num_bound=200000/group_len
    lower_bound=300
    
    extracted_attrs=[keys[i] for i, j in enumerate(values) if j > lower_bound]
    

    print(lower_bound)
    extracted_group=group_count.iloc[:len(extracted_attrs)] 
    selected_vector=[]
    
    for i in extracted_group.index:
         selected_vector.append(list(i))
    selected_vector=pd.DataFrame(selected_vector, columns=names)
    print(selected_vector.head())
    cnt=selected_vector.sum()
    print(cnt)
    for index, i in enumerate(cnt):
        if i==selected_vector.shape[0] or i==0:
            print(names[index])
        
        
    # print(selected_vector.head())
    # one=[1]*selected_vector.shape[1]
    # zero=[0]*selected_vector.shape[1]
    # selected_vector=np.array(selected_vector)
    # selected_vector=selected_vector.transpose()
    # for i in selected_vector:
    #     if list(i) 
    
    # selected_vector=pd.DataFrame(extracted_group.index)
    
    # print(selected_vector.head())
    
    # for i in selected_vector.columns:
    #     print(selected_vector[i])
    #     print(type(selected_vector[i]))
    
    #=keys[len(extracted_attrs)]
    
    extracted_group.to_csv(f"./{fix}_group_number.csv")
    
    print(len(extracted_attrs))
    
    
    colors=[]
    for i in values:# 
        if i >expected_num_bound:
            colors.append("navy")
        else:
            colors.append("blue")    
            
    ## number limitation
    X_axis=X_axis[:limit]
    values=values[:limit]
    X=X[:limit]      

           
           
           
    #print(extracted_attrs)        
        
    plt.bar(X_axis - 0.4, values, 0.4, label = 'The number of the group', color=colors)
    #plt.bar(X_axis - 0.2, y[1], 0.4, label = 'Service', color=service_color)

    plt.xticks(X_axis, X)
    plt.xlabel("The Partition of the possible attrs")
    plt.ylabel("The number of the group")
    plt.title(f"{fix}: Domain Distribution/ expected num bound: {expected_num_bound}")
    plt.legend()
    plt.show()

for i in range(len(fixed_attrs)):
    service_dataset_distribution(ATTRIBUTE_FILE, chosen=attibutes_by_fixed[i], fix=fixed_attrs[i], hair_priority=hair_priority) 
    
    
    # print(ann_file[service_attribute_names].value_counts().keys()[:50])
    # print(ann_file[service_attribute_names].value_counts().values[:50])
    
    # print(ann_file[reverse_attribute_names].value_counts().keys()[:50])
    # print(ann_file[reverse_attribute_names].value_counts().values[:50])
    
    # print(len(ann_file[reverse_attribute_names].value_counts().keys()))
   
    
    #reverse_keys=[for i in ann]
    
    # keys=ann_file[service_attribute_names].value_counts.keys()

    # dict.from_keys()

    # dict={key:value for key, value in ann_file[service_attribute_names].value_counts()}
    
    # print(dict)
    
    #service_attr=ann_file[service_attribute_names]
 
#service_dataset_distribution(ATTRIBUTE_FILE, "all", 100)    
#hair_color_anomaly(ATTRIBUTE_FILE)   
    
    # hair_color=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    

    # for i in range(c_dim):

  
    #     if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
    #         c_trg[:, i] = 1
    #         for j in hair_color_indices:
    #             if j != i:
    #                 c_trg[:, j] = 0
    #     else:
    #         # print(i)
            
    #         c_trg[:, i] = (c_trg[:, i] == 0)  # Revoerse attribute value.
#proportion_chk(ATTRIBUTE_FILE)