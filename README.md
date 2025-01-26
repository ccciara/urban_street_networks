# Evaluating Urban Accessibility: Comparative Representations of Street Network Topology Using GNNs
This project aimed to create information-dense embeddings of street network patterns through supervised and unsupervised GNNs, and compare their accuracy against traditional street network measurements (average nodal degree) in predicting qualities of their internal environment. 

## Data
Data consisted of street networks of Medium Super Output Areas (MSOAs) within urban and suburban areas of England. Each street network graph was paired with a value representing the number of amenities (schools, shops, etc) within 15 minutes walk for internal residents. Rural areas were identified with the Spatial Signature dataset by Urban Grammar Project, which used satellite imagery to label UK environment types, and dropped from the dataset. Although I analysed all of England, we'll focus on just a few cities in data visualisations to show more detail. Here's the ground truth for amenity accessibility in four major cities.

![download](https://github.com/user-attachments/assets/02eea018-0843-4730-9ba5-7d4451afb97c)

Several iterations of graphs were tested with different node attributes incorporated, including neutral coordinates (distance of each node from center of graph) and Laplacian-based positional encoding (information on the relationship of each node with the larger graph. Giving each node a scalar value of 1 ended up being most effective, as this gives each node some data to pass during message passing layers.


## 1. Training Predictive Model
The below method was used to train and test various Graph Neural Network model architectures to determine the optimal message-passing layer and hyper parameters.

<img width="682" alt="Screenshot 2025-01-24 at 8 23 34 AM" src="https://github.com/user-attachments/assets/386ea3ac-2b80-44ac-a3d3-b843b5a46c7b" />


4 different layer types were tested; Graph Isomorphic Network (GIN), Graph Convolutional Network (GCN), Graph Sample and Aggregation (GraphSAGE), and Dynamic Graph Attention Network (GATv2). Of these, GIN emerged as the most successful in predicting amenity counts. Hyperparameters like number of message-passing layers and hidden channels were also determined here. Below you can see the predictions of the most successful model, a 3 layer GIN model with 64 hidden channels.

![download-2](https://github.com/user-attachments/assets/11ac56d1-d8ab-4889-a331-fc16f452f508)

Before finishing up here, a spatial dependency test was run to make sure that neighbouring graphs were not leaking data from the training set to the test set. The model was trained on everything outside of the West Midlands and then tested only on the West Midlands. Its performance did not decline, indicating that the model was not relying on any data leakage.

Before moving on, the model was trained until loss plateaued and frozen in that state. All graphs were passed through, but rather than returning predictions, the embeddings were extracted just before the final stage.

### 1.2 Testing GNNExplainer
Explainability is a major issue in deep learning as it is often difficult to determine the basis on which models make their predictions. I tried to tackle that here with a GNNExplainer intended to interpret the reasoning behind the predictions made by my predictive model. The GNNExplainer works by optimising a mask over the input features and the graph structure, isolating selective parts of the graph and measuring which information is necessary for the model to reach the same prediction as it would with the entire graph. What I was really hoping for here was the illumination of particular patterns associated with high or low amenity accessibility, which could good knowledge for building or rebuilding accessible areas. Unfortunately, the results seemed random and hard to interpret, so I'm not sure if I made an error somewhere or if the GNNExplainer module was unable to capture defining patterns and features.


## 2. Training Autoencoder
This method used a non-supervised approach to creating embeddings. Graph autoencoders are composed of two units; an encoder which embeds graph data into a dense lower dimensional space, and a decoder which attempts to reconstruct the graph from the embedding information. I used the same parameters and hyperparameters as my most successful predictive model, and once the autoencoder training plateaued I froze the model and passed all the graphs through the encoder, extracting the embeddings.

### 2.2 Examining Autoencoder Results
The embeddings were simplified further with PCA and clustered with K-means in order to visualize patterns in street network types and their spatial distribution. Four cluster types were identified, with clear visual differences between them in regards to patterns and complexity. Here's examples of Cluster 1 and 4. (I stole these from my GNNExplainer visualizations, and unfortunately don't have examples on hand of 2 and 3, and no longer have all the files on my computer to create them! You can also see here how the nodes and edges indicated by the GNNExplainer to be important are not very interpretable.)

MSOA E02000730 in Newham, high predicted amenity walkability explanation (Cluster 1)
<img width="839" alt="Screenshot 2025-01-25 at 4 51 13 PM" src="https://github.com/user-attachments/assets/7d98be10-ab4b-4c2f-a238-40997e41468c" />

MSOA E02001353 in Liverpool, low predicted amenity walkability explanation (Cluster 4)
<img width="841" alt="Screenshot 2025-01-25 at 4 50 32 PM" src="https://github.com/user-attachments/assets/257e646f-78c6-4732-b8ac-f72bac3714b1" />

Visualizing these cluster types across several cities shows how cluster types are spatially autocorrelated, which confirms that the embeddings are effective in retaining topological data as no geospatial coordinates were input.

![download-3](https://github.com/user-attachments/assets/a8d2eefa-b132-44a0-8d6e-f5bf31512383)

Looking at the qualities of each cluster also shows some interesting patterns! Though I expected more interconnected areas to have more amenities, as these patterns usually occur in more urban areas, Cluster 1 which ranked 2nd in interconnection (represented as average nodal degree) had the best amenity walkability. Though these Cluster 1 areas had lower street density, they were the most densely populated, showing the importance of population density in predicting amenity availability.

<img width="626" alt="Screenshot 2025-01-25 at 4 59 02 PM" src="https://github.com/user-attachments/assets/b3ca7f90-d77a-40fe-b846-58f49ad4873c" />


## 3. Spatial Error Modelling
Finally, the two sets of embeddings were compared against each other and also against average nodal degree in their capabilities to predict amenity walkability. Average nodal degree is the typical measurement used to summarize street network patterns; it is the average number of edges (streets) connected to each node (intersection) within the street network. Spatial autocorrelation was first confirmed with a linear regression and Global Moran's I test upon the residuals. Spatial lag model or geographically weighted regression could have also been used instead of spatial error model, but I decided to go with spatial error model under the logic that variables not accounted for, for example population density, likely were playing a large role in the error term.

Results of the Spatial Error Model showed that the embeddings explained 1.3x more variance in walkable amenity counts than average nodal degree. The prediction-trained embeddings received a pseudo R-squared score of 0.409, the autoencoder embeddings 0.410, and the nodal degree 0.317 (For the full results tables, open my paper!). All of these scores, which measure how much variance in amenity walkability can be explained with the street network measurement, are definitely pretty low. There is a lot more that goes into amenity location than street network patterns, for example, population density, street network density, and socioeconomic qualities. However, I think this is still a pretty good indication that the extra data in the embeddings is more useful than average nodal degree - and that more information incorporated into these graphs and embeddings could be effective in future analysis.


## 4. Next Steps
Future goals for this project revolve around increasing the amount of information within the graphs and embeddings for more experimental analysis. The geographical units need to be standardized for each graph; this will be done with h3 tiles. Ideally I hope to combine walking, biking and driving routes within each graph, labeling route types with edge attributes - this was partially attempted during this project but difficulties with models dealing with edge attributes led to dropping it in the interest of getting the project done.

The sky is the limit in terms of other kinds of information that could potentially be included; environmental, such as presence of trees or green spaces; social, such as locations of different amenity types like schools, stores, or housing; economic, such as house price or regional economic output; or even demographic, such as the age, professions or backgrounds of residents.

In my opinion the autoencoder stage with clustering was the most interesting and informative part of this research, and I would be interested in further clustering more information-dense embeddings to create better understandings of neighbourhoods and divisions. However, I think it would also be possible and informative to apply these embeddings towards other social / economic / environmental questions, to reach more environmentally contextual understandings.
