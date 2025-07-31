import pandas as pd
import ollama
from tqdm import tqdm
import time

dataset = 'baby'

interactions = pd.read_csv('../data/{}/{}.inter'.format(dataset, dataset), skiprows=1, sep='\t', usecols=[0,1], names=['userID', 'itemID'])
interactions = interactions.rename(columns={'ItemID': 'itemID'})

df = pd.read_csv('../data/{}/output_image2text.csv'.format(dataset))

i_id, desc_str = 'itemID', 'description'

df[desc_str] = df[desc_str].fillna(" ")
df['title'] = df['title'].fillna(" ")
df['brand'] = df['brand'].fillna(" ")
df['categories'] = df['categories'].fillna(" ")
df['image_description'] = df['image_description'].fillna(" ")

sentences = []
for i, row in df.iterrows():
    sen = row['title'] + ' ' + row['brand'] + ' '
    cates = eval(row['categories'])
    if isinstance(cates, list):
        for c in cates[0]:
            sen = sen + c + ' '
    sen += row[desc_str]
    sen += row['image_description']
    sen = sen.replace('\n', ' ')

    sentences.append(sen)

df['combined_description'] = sentences

df['itemID'] = df['itemID'].astype(int)
user_items = pd.merge(interactions, df[['itemID', 'combined_description']], on='itemID', how='left')
user_items.sort_values(by=['userID'], inplace=True)

user_descriptions = user_items.groupby('userID')['combined_description'].apply(list).reset_index()

prompt_template = "Please reason about the user preferences based on the following list of item descriptions that he or she has interacted with. The list is: {descriptions}. To generate the user preferences using a one-paragraph natural language in no more than 100 words."

def generate_user_preference(descriptions_list):
    if not descriptions_list:
        return "无法确定偏好 - 无足够交互数据"
    
    
    prompt = prompt_template.format(descriptions=descriptions_list)
    
    try:
        response = ollama.generate(
            model='gemma3:27b',
            prompt=prompt,
            # options={'temperature': 0.3}
        )
        return response['response'].strip()
    except Exception as e:
        print(f"Error generating preference: {e}")
        return None
    
user_preferences = []
batch_size = 10

for i in tqdm(range(0, len(user_descriptions), batch_size)):
    batch = user_descriptions.iloc[i:i+batch_size]
    
    batch_results = []
    for idx, row in batch.iterrows():
        preference = generate_user_preference(row['combined_description'])
        print(f"行 {idx}: generate_user_preference: {preference}")
        batch_results.append({
            'userID': row['userID'],
            'preference': preference,
            'num_interactions': len(row['combined_description'])
        })
    
    user_preferences.extend(batch_results)
    
    pd.DataFrame(user_preferences).to_csv('../data/{}/user_preferences.csv'.format(dataset), index=False)

pd.DataFrame(user_preferences).to_csv('../data/{}/user_preferences.csv'.format(dataset), index=False)
print("User preference generation completed!")