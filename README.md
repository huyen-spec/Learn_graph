# Learn_graph

Graph basics:


# Graph data:
+ Nodes
+ Egdes
+ Adjacency matrix
+ Bipartite graph: A graph that can be seperated into 2 independent sets where there is no link between the vertex that nelong to the same half.

# Graph neural networks:
+ Graph convolution network GCN:
Formula:
      H(l+1)=σ(D^−1/2A^D^−1/2H(l)W(l))
    D the adjacency matrix and W the weight of convolution applied on the nodes. Just a matrix multiplication to get sum of neighboring nodes (the non-related nodes have index ) in the adjacency matrix) after the node is linearly transformed.
https://mlabonne.github.io/blog/intrognn/
+ GraphSage: Use sampling techniques to obtain the most important nodes for aggregation. 
https://mlabonne.github.io/blog/graphsage/
+ Graph Attention Network: Use attention mechanism to compute the weights of aggregation among nodes.
https://mlabonne.github.io/blog/gat/
+ GIN:
Weisfeiler-Lehman test: test if the graphs are isomorphic by calculating the sum of neighboring nodes of a node: https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/
Link: https://towardsdatascience.com/how-to-design-the-most-powerful-graph-neural-network-3d18b07a6e66
https://blog.csdn.net/m0_46306014/article/details/118498341
https://mlabonne.github.io/blog/graphsage/


# Implementation of GCN
- Notebooks:
  Implementation of GCN and GAT tutorial with examples: https://github.com/huyen-spec/Learn_graph/blob/main/GNN_overview.ipynb
  (their tutorial very clear)
- Torch geometric library:
  check this link: https://zqfang.github.io/2021-08-07-graph-pyg/
  def forward(self, x, edge_index):
      # x has shape [num_nodes, in_channels]
      # edge_index has shape [2, E]
  x: input, ex: x:[10,200] -> there are 10 nodes, each with dim 200
  edge_index: ex:[2, 1000]: there are 1000 edges (edges different from num of nodes), first row the index of source node and second row target node -> the node edge_index[0][i] is connected to edge_index[1][i]
  
  def message(self, x_j, edge_index, size):
    #x_j has shape [num_edges, out_channels]
    
  obtained x and edge_index, we can get x_j = (x[edge_index[0]] or x[edge_index[1]]) which is the souce node's features
  
  Check out this example here: https://github.com/huyen-spec/Learn_graph/blob/main/graph_examples/graph0606.py

# Constructing a bipartite graph from prescriptions
  we count the coocurrences of pills in the presciption and then indirectly construct pill-pill relationships based on the NLP formula:
  ![image](https://user-images.githubusercontent.com/62921312/176898061-418c84b0-1a56-458c-8308-915c71ee4dc1.png)


# About Apriori
An association mining problem can be decomposed into two subproblems:

Find all combinations of items in a set of transactions that occur with a specified minimum frequency. These combinations are called frequent itemsets. (See "Frequent Itemsets" for an example.)

Calculate rules that express the probable co-occurrence of items within frequent itemsets. (See "Example: Calculating Rules from Frequent Itemsets".)

Apriori calculates the probability of an item being present in a frequent itemset, given that another item or items is present.

Association rule mining is not recommended for finding associations involving rare events in problem domains with a large number of items. Apriori discovers patterns with frequency above the minimum support threshold. Therefore, in order to find associations involving rare events, the algorithm must run with very low minimum support values. However, doing so could potentially explode the number of enumerated itemsets, especially in cases with a large number of items. This could increase the execution time significantly. Classification or anomaly detection may be more suitable for discovering rare events when the data has a high number of attributes.

# Association Rules
The Apriori algorithm calculates rules that express probabilistic relationships between items in frequent itemsets For example, a rule derived from frequent itemsets containing A, B, and C might state that if A and B are included in a transaction, then C is likely to also be included.

An association rule states that an item or group of items implies the presence of another item with some probability. Unlike decision tree rules, which predict a target, association rules simply express correlation.

# Antecedent and Consequent
The IF component of an association rule is known as the antecedent. The THEN component is known as the consequent. The antecedent and the consequent are disjoint; they have no items in common.

Oracle Data Mining supports association rules that have one or more items in the antecedent and a single item in the consequent.

# Confidence
Rules have an associated confidence, which is the conditional probability that the consequent will occur given the occurrence of the antecedent.

# Metrics for Association Rules
Minimum support and confidence are used to influence the build of an association model. Support and confidence are also the primary metrics for evaluating the quality of the rules generated by the model. Additionally, Oracle Data Mining supports lift for association rules. These statistical measures can be used to rank the rules and hence the usefulness of the predictions.

# Support
The support of a rule indicates how frequently the items in the rule occur together. For example, cereal and milk might appear together in 40% of the transactions. If so, the following two rules would each have a support of 40%.

cereal implies milk
milk implies cereal
Support is the ratio of transactions that include all the items in the antecedent and consequent to the number of total transactions.

Support can be expressed in probability notation as follows.

support(A implies B) = P(A, B)

# Confidence
The confidence of a rule indicates the probability of both the antecedent and the consequent appearing in the same transaction. Confidence is the conditional probability of the consequent given the antecedent. For example, cereal might appear in 50 transactions; 40 of the 50 might also include milk. The rule confidence would be:

cereal implies milk with 80% confidence
Confidence is the ratio of the rule support to the number of transactions that include the antecedent.

Confidence can be expressed in probability notation as follows.

confidence (A implies B) = P (B/A), which is equal to P(A, B) / P(A)

# Lift
Both support and confidence must be used to determine if a rule is valid. However, there are times when both of these measures may be high, and yet still produce a rule that is not useful. For example:

Convenience store customers who buy orange juice also buy milk with 
a 75% confidence. 
The combination of milk and orange juice has a support of 30%.
This at first sounds like an excellent rule, and in most cases, it would be. It has high confidence and high support. However, what if convenience store customers in general buy milk 90% of the time? In that case, orange juice customers are actually less likely to buy milk than customers in general.

A third measure is needed to evaluate the quality of the rule. Lift indicates the strength of a rule over the random co-occurrence of the antecedent and the consequent, given their individual support. It provides information about the improvement, the increase in probability of the consequent given the antecedent. Lift is defined as follows.

(Rule Support) /(Support(Antecedent) * Support(Consequent))
This can also be defined as the confidence of the combination of items divided by the support of the consequent. So in our milk example, assuming that 40% of the customers buy orange juice, the improvement would be:

30% / (40% * 90%)
which is 0.83 – an improvement of less than 1.

Any rule with an improvement of less than 1 does not indicate a real cross-selling opportunity, no matter how high its support and confidence, because it actually offers less ability to predict a purchase than does random chance.

# Link: 
https://docs.oracle.com/cd/E18283_01/datamine.112/e16808/algo_apriori.htm#:~:text=generate%20fewer%20rules.-,Metrics%20for%20Association%20Rules,supports%20lift%20for%20association%20rules





# Scene graph generation:




