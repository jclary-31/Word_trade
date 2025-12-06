# Word_trade
An exploration of word trade through power BI ;  and neural network (in python) to better understand relationships[not finished] \
The code in main_nb can be adapted for any country, with 3 letter ISO format\
The SITC Classification is : \
0 Food and live animals \
2 Crude inedible materials , no fuels \
3 mineral and fuels \
5 Chemical and related (e.g. medicinal, fertilizers, plastics) \
6 manufactured goods \
8 miscellaneous manufactured aricles (not in 6) \
TOTAL all other combined


# visualizatopm with power BI
<p align="center">
 <img width="400" src=Dashboard1.png>
 </p>

# example of graph vizualisation
balance of all goods, each country is connected to 4 other countries (top2 importer+ top2 exporter). Between the starting country (here France), and endpoints, there are 3 connections max.
<p align="center">
 <img width="600" src=NN_allgoods_balance_fromFrance.png>
 </p>


# Country dependance for a given class of product
balance of all goods, with a network starting at France and looking for the top 4 strongest exchanges (abs(balance) here, but it could be import+export)
<p align="center">
 <img width="600" src=NN_allgoods_balance_France.png>
 </p>
percentage in black mean positive balance from A to B, negative is in red. For example FRA->DEU is 4% in red, meaning that France have a negative balance with Germany, while Germany have a 8% in black, meaning that Germany have a positive balance with France.\
This example also show that France is twice as important for Germany than Germany for france. This non symetrical importance made me look for a way to compute other countries importances

From this network, I also computed the each country relative importance by using a weighted version of Pagerank algorithm. This algorithm is the origin of web page appearance in web search.  I adapted the description existing here https://cs50.harvard.edu/ai/projects/2/pagerank/. The weights, i.e. 'out' edges relative strength and endnodes are not in the aformentionned link. So I did how I though was correct but some test would be nice for confirmation/correction.
For the network in the example before, I have : \
CHN 14.96 \
USA 12.51 \
DEU 10.15 \
HKG 9.44 \
FRA 8.67 \
GBR 8.17 \
NLD 7.64 \
IND 7.39 \
NOR 7.14 \
ITA 6.97 \
BEL 6.96 \

Some comments here, 1) most important country for France is... China, 2) France and USA are not directly connected but USA is the 2nd most important country to France exchanges; and 3) I don't know how to interpret France in this rank. Maybe I should juste remove it and renomralize but it feels odd. 

