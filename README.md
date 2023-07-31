# 1.methodology part. 

## 1.1 What is a image retrieval system ?
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/f9cc7182-8fd8-4f12-9978-679e6345a06e" />
</p>
<p>
The mechanism of Image_retrieval operates as follows: First, the characteristic features of the search target are obtained in advance and stored in a database. When a user performs a search using a query image, the features of the query image are also extracted. Then, the extracted features of the query image is compared with the features stored in the database for each image, and the images with high similarity are ranked. Finally, the top-ranked images are presented to the user.In this project, I used the fashion items' datasets.
</p>

## 1.2 DeepFashion Dataset
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/4d598830-abe6-4a07-a8e3-5d5cc2841d9a" />
</p>
I used the DeepFashion dataset(https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/Consumer2ShopRetrieval.html), which contains a variety of consumer-to-shop clothes  images. In this project, a total of 226,980 images were used. These images include both customer and shop images. Furthermore, this dataset can be organized into three categories: "large categories," "small categories," and "IDs." The big categories include DRESSES, TOPS, and TROUSERS. The small categories represent more specific types of fashion items, and the IDs mean individual products. 
<p><b>note:You need to get permission from the creator if you want to use this dataset.</b></p>

## 1.3 What is a deep metric learning?
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/3cc779b1-8167-4670-95f4-33537f5b7984" />
</p>
My proposal system was developed using a type of Deep Metric Learning called "triplet loss." Triplet loss aims to learn a metric where similar images  are close to each other, and dissimilar images are far apart. In this method, the anchor represents a customer's image. Positive is the same product as ANCHOR and is the store image. the negative represents an image from the same small category but it is different product from the anchor and positive. In other words, the negative is somewhat close to the anchor but not the same product. For example, in the case of images, the anchor and positive are both tank tops of the same design, while the negative is also a tank top but of a different product (ID). This kind of relationship is commonly referred to as "hard negative." To achieve this, the difference of feature vectors in anchor and positive and the difference of feature vectors between anchor and negative,  are calculated using the Euclidean distance as the triplet loss function.

# 2.Code part
