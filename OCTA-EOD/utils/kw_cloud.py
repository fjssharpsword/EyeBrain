from wordcloud import WordCloud
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

#print(matplotlib.matplotlib_fname())

#https://amueller.github.io/word_cloud/auto_examples/colored.html
text = ''
with open('/data/pycode/EyeBrain/utils/patent2021.csv', encoding="GBK") as f:
    for line in f:
        items = line.split(';')
        for item in items:
            item = item.replace('\n', '') #replace line break
            item = item.replace(' ', '') #replace space
            text = text + " " + item

# square
#wc = WordCloud(font_path="/data/pycode/EyeBrain/utils/simkai.ttf", background_color="white",width=1000, height=860, margin=2).generate(text)
#wc.to_file("/data/pycode/EyeBrain/imgs/patent_square.png") 

#round shape
x, y = np.ogrid[:900, :900]
mask = (x - 450) ** 2 + (y - 450) ** 2 > 430 ** 2
mask = 255 * mask.astype(int)

wc = WordCloud(font_path="/data/pycode/EyeBrain/utils/simkai.ttf", background_color="white", repeat=True, mask=mask)
wc.generate(text)
wc.to_file("/data/pycode/EyeBrain/imgs/patent_round.png") 
#plt.axis("off")
#plt.imshow(wc, interpolation="bilinear")
#plt.savefig('/data/pycode/EyeBrain/imgs/patent_round.png', dpi=100)