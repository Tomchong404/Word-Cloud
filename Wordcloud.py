#!/usr/bin/env python
# coding: utf-8

# In[1]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import numpy as np
from PIL import Image
from nltk.corpus import stopwords
from nltk import bigrams
from nltk import FreqDist
from nltk import ngrams
from PyPDF2 import PdfReader 


# In[2]:


reader = PdfReader('NWS AR 2023.PDF')
total_pages = len(reader.pages)
print(total_pages)

for page_number in range(72, 95):
    page = reader.pages[page_number - 1]
    text = page.extract_text()
    print(text)


# In[16]:


# Download stopwords from NLTK
nltk.download('stopwords')

# Additional stopwords
stopwords_list = set(stopwords.words('english') + ['view','holdings','advisers','discount','may','indonesia','total','certify','nm','rating','base','finance','next','price','to', 'etc', 'ie', 'andor', 'xd', 'due', 'eg', 'lack', 'inadequate', 'insufficient', 'risk','clsa','hong','securities','research','report','pte','eps''certify','kong','profit','attributable','banking','disclosures','contact','na','please','refer','private','sdn','herein','pty','registration','india','subject'])

# Convert to lowercase
text_lower = text.lower()
# Remove stopwords
split_text = ' '.join([word for word in nltk.word_tokenize(text_lower) if word.isalpha() and word not in stopwords_list])

# Create WordCloud
wordcloud = WordCloud(width=1000, height=500, background_color="white", collocations=False).generate(split_text.title())

# Display WordCloud
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[13]:


# Download stopwords from NLTK
nltk.download('stopwords')

# Additional stopwords
stopwords_list = set(stopwords.words('english') + ['remains','may','difficulty','recovery','report','to', 'etc', 'ie', 'andor', 'xd', 'due', 'eg', 'risk','could','requires','affect','including','also','holdings'])

# Convert to lowercase
text_lower = text.lower()

# Create bigrams and remove stopwords
bigrams_list = [' '.join(b) for b in bigrams(nltk.word_tokenize(text_lower)) if all(word.isalpha() and word not in stopwords_list for word in b)]

# Create a frequency distribution of bigrams
bigrams_freq = FreqDist(bigrams_list)

# Create WordCloud based on frequency distribution
wordcloud1 = WordCloud(width=1000, height=500, background_color="white", collocations=True).generate_from_frequencies({k.title(): v for k, v in bigrams_freq.items()})

# Display WordCloud
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()


# In[15]:


from PIL import Image
import PIL.Image
silhouette_mask = np.array(PIL.Image.open('NWS logo.png'))
wordcloud3 = WordCloud(mask=silhouette_mask, background_color="white", contour_color="black").generate(split_text)
# Display WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud3)
plt.axis('off')
plt.show()

