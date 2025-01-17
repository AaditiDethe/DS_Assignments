
############### Association Rules Assignment ##############################

#1.
'''Problem Statement: -
Kitabi Duniya, a famous book store in India, which was established before Independence, 
the growth of the company was incremental year by year, but due to online selling of books
 and wide spread Internet access its annual growth started to collapse, seeing sharp
 downfalls, you as a Data Scientist help this heritage book store gain its popularity back 
 and increase footfall of customers and provide ways the business can improve exponentially,
 apply Association RuleAlgorithm, explain the rules, and visualize the graphs for clear
 understanding of solution.
1.) Books.csv
'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("C:/Association_Rules/books.csv",on_bad_lines='skip')
df

transactions = df.applymap(str).values.tolist()
te = TransactionEncoder()

te_ary = te.fit(transactions).transform(transactions)
df1 = pd.DataFrame(te_ary, columns = te.columns_)
df1

frequent_itemsets = apriori(df1,min_support = 0.01,use_colnames = True)
frequent_itemsets

rules = association_rules(frequent_itemsets,metric = 'confidence', min_threshold = 0.7)
rules 

####################################################################################

#2.
'''Problem Statement: - 
The Departmental Store, has gathered the data of the products it sells on a Daily basis.
Using Association Rules concepts, provide the insights on the rules and the plots.
2.) Groceries.csv
'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("C:/Association_Rules/Groceries.csv",on_bad_lines='skip')
df

transactions = df.applymap(str).values.tolist()
te = TransactionEncoder()

te_ary = te.fit(transactions).transform(transactions)
df1 = pd.DataFrame(te_ary, columns = te.columns_)

df1

frequent_itemsets = apriori(df1,min_support = 0.4,use_colnames = True)
frequent_itemsets

rules = association_rules(frequent_itemsets,metric = 'confidence', min_threshold = 0.7)
rules

##################################################################################

#3.
'''Problem Statement: - 
A film distribution company wants to target audience based on their likes and dislikes,
 you as a Chief Data Scientist Analyze the data and come up with different rules of movie
 list so that the business objective is achieved.
3.) my_movies.csv

'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("C:/Association_Rules/my_movies.csv")
df

te = TransactionEncoder()
te_ary = te.fit(df).transform(df)
df1 = pd.DataFrame(te_ary, columns = te.columns_)

df1

frequent_itemsets = apriori(df1,min_support = 0.4,use_colnames = True)
frequent_itemsets

rules = association_rules(frequent_itemsets,metric = 'lift', min_threshold = 0.6)
rules

###################################################################################

#4.
'''Problem Statement: - 
A Mobile Phone manufacturing company wants to launch its three brand new phone into the 
market, but before going with its traditional marketing approach this time it want to 
analyze the data of its previous model sales in different regions and you have been hired
 as an Data Scientist to help them out, use the Association rules concept and provide your
 insights to the company’s marketing team to improve its sales.
4.) myphonedata.csv
'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("C:/Association_Rules/myphone.csv", on_bad_lines='skip')
df

te = TransactionEncoder()
te_ary = te.fit(df).transform(df)
df1 = pd.DataFrame(te_ary, columns = te.columns_)

df1

frequent_itemsets = apriori(df1,min_support = 0.01,use_colnames = True)
frequent_itemsets

rules = association_rules(frequent_itemsets,metric = 'lift', min_threshold = 0.6)
rules

##################################################################################

# 5
'''Problem Statement: - 
A retail store in India, has its transaction data, and it would like to know the buying
 pattern of the consumers in its locality, you have been assigned this task to provide the
 manager with rules on how the placement of products needs to be there in shelves so that 
 it can improvethe buyingpatterns of consumes and increase customer footfall. 
 5.) transaction_retail.csv
'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("C:/Association_Rules/transactions_retail.csv")
df  
df.dropna(inplace=True)

transactions = df.applymap(str).values.tolist()
te = TransactionEncoder()

te_ary = te.fit(transactions).transform(transactions)
df1 = pd.DataFrame(te_ary, columns = te.columns_)
df1

frequent_itemsets = apriori(df1,min_support = 0.01,use_colnames = True)
frequent_itemsets

rules = association_rules(frequent_itemsets,metric = 'lift', min_threshold = 1)
rules

'''Business Objectives:
    1. Drive more in-store purchases by analyzing customer purchasing behavior and identifying trends.
    2. Identify frequently bought together items to promote bundle sales and upselling opportunities.
    3. Use data to suggest movies that align with individual preferences, thereby enhancing the customer experience and increasing the likelihood of movie rentals or purchases.
    4. Understand buying patterns across different regions to customize marketing strategies and product features according to regional preferences.
    5. Identify frequently bought together items and place them closer on shelves to encourage impulse buying and convenience for shoppers.
    '''