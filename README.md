# Word_trade
An exploration of word trade through power BI ;  and neural network (in python) to better understand relationships[not finished]

# example with power BI
<p align="center">
 <img width="800" src=Dashboard1.png>
 </p>

# example of graph vizualisation
balance of all goods, each country is connected to 4 other countries (top2 importer+ top2 exporter). Between the starting country (here France), and endpoints, there are 3 connections max.
<p align="center">
 <img width="800" src=NN_allgoods_balance_fromFrance.png>
 </p>


# Country dependance for a given class of product
balance of all goods, with a network starting at France and looking for the top 4 strongest exchanges (abs(balance) here, but it could be import+export)
<p align="center">
 <img width="800" src=NN_allgoods_balance_France.png>
 </p>


From this network, I also computed the each country relative importance by using a weighted version of Pagerank algorithm. This algorithm is the origin of web page appearance in web search.  I adapted the description existing here https://cs50.harvard.edu/ai/projects/2/pagerank/. The weights, i.e. 'out' edges relative strength and endnodes are not in the aformentionned link. So I did how I though was correct but some test would be nice for confirmation/correction.
For the network in the example before, I have :
CHN 14.96
USA 12.51
DEU 10.15
HKG 9.44
FRA 8.67
GBR 8.17
NLD 7.64
IND 7.39
NOR 7.14
ITA 6.97
BEL 6.96
