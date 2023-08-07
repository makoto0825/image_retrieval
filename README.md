# 1.methodology part. 
In this part, I will explain about some methodology or notions in my proposal system.

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
In this section, I will show the code and explain it. My Deep metric learning program was developed with reference to following two sites.
<ul>
  <li>Keras(https://keras.io/examples/vision/siamese_network/)</li>
  <li>kaggle(https://www.kaggle.com/code/xhlulu/shopee-siamese-resnet-50-with-triplet-loss-on-tpu)</li>
</ul>

## 2.1 Development Environment
I used Google Colaboratory as our Integrated Development Environment (IDE). The reason for this choice is that Colaboratory provides high-performance GPUs, which are beneficial for training deep learning models. Specifically, I utilized the NVIDIA A100 GPU for our tasks. For programming, I opted to use Python and selected Keras and Tensorflow as our deep learning libraries for training.

## 2.2 Flow of the program
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/ca95e497-ad0e-46b2-b47d-dfbfa163bd48" />
</p>
The flow of the program is as follows:
1.	First, I create the dataset from the training images, generating data frames for both training (df_train) and validation (df_val) purposes. These data frames contain the file paths of the images.
2.	Next, I define the architecture to be used for training.
3.	I perform the training using the defined architecture and obtain the feature vectors for both the query image and the database images using the trained architecture.
4.	Finally, I compare the similarity of the feature vectors to conduct image retrieval.

## 2.3 Create training data.
First, I put some folders which are DRESSES,TOPS and TROUSERS at Google colaboratory from deep fashion datasets. And I stored these three folders in a folder called "img". The The reason why I did not put them at Google drive, it takes long time to read those images when training the model. I definitely recommend you to put them in Google colaboratory, not Google drive. Moreover, I made a data frame which describes each image file path by reading "img" folder. 
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/e2ed4819-eb61-4ea4-91a3-929ca0b78610" />
</p>

Next,I added new label for small categories. This label was used for making triplet pair of images later.
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/a938e7f5-3206-4ae1-b052-1e3b9cae3555" />
</p>
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/faa211db-6bce-4f77-acfd-634ae8f8da35" />
</p>

I made test_df, train_df, validation_df from df2. Which image belongs to train, test, or validation is determined from the information in the list_eval_partition.xlsx file.　This xlsx file was made myself. Hence you need make it yourself from list_eval_partition.txt in deepfashion. This text file shows the combination of anchor and positive, and which type it belongs to: test, train, or validation.
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/43377ad8-7fd1-46e3-80b4-363147201e8e" />
</p>

I then merge the "kind" information (train, val, test) from the pre-prepared Excel file and add it to img_df2. Using the "kind" column as a filter, I create separate dataframes for training, validation, and testing. However, the image paths (in the "anchor" column) in the created dataframes contain both customer and shop images. To address this, I remove all rows that contain the string "shop_" in the path. I then create a new column called "positive" based on the Excel file's information about the matching of anchor and positive images. 
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/ead9ba63-c321-4a27-af93-a81bdaee9e47" />
</p>

Following  image is sample of train_df, which has several imformation  such as anchor,id, small category, positive, kind.
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/d5616c97-49f6-44d4-9d30-1b9536328f31" />
</p>

Once the dataframes are created, the next step is to build TensorFlow datasets from the df. First, I define a function for image preprocessing. Specifically, the function reads the image files, resizes them, and applies normalization preprocessing for ResNet-style processing. 
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/48825d66-886c-4756-b8f2-b73a8e61a5bd" />
</p>
Anchor and positive images are obtained directly from the "anchor" and "positive" columns of the created df, respectively. On the other hand, negative images are randomly selected from the paths of images where the 'Label1 (small category)' column is equal to the anchor but with different IDs (hard negative).
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/801fb273-b50f-45da-bee4-33322d12fc3b" />
</p>

Once the functions are defined, I use the create_triplets function and the previously created df to generate lists of triplets. Then, using this information, I create TensorFlow datasets. The created datasets are batched and optimized for faster processing with prefetching. Additionally, the data order is randomized. This process is carried out for both train_df and vali_df datasets.
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/088f2c95-6ce5-4b30-906f-ce78a5908716" />
</p>

## 2.4 Deep learning.
First, I define a custom Keras layer called DistanceLayer() to perform distance calculations. This layer is used to compute the distances between anchor and positive images, as well as between anchor and negative images. It is intended for calculating the triplet loss.
<p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/87a82672-38f3-45c4-b0c8-4d36d1fc3267" />
</p>
 Next, I create an architecture by adding several layers to the baseline model ResNet50 (Figure 56). Dropout layers were added to prevent overfitting, and a dense layer with 512 dimensions was added as the output layer. The feature vectors output from this layer are used to compute distances using the previously defined DistanceLayer. The training is restricted to layers below stage 5.
 <p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/93f0868e-b0f6-4d6a-a3ac-b1e9a8adb028" />
</p>

 I then define the class SiameseModel, which implements triplets along with custom training and testing loops. The train_step(self, data) method, defined within the class, sets up a custom training loop. It takes one batch during training, computes the loss for that batch, and updates the model weights. The test_step(self, data) method defines a custom validation loop. It takes one batch (data) during validation, computes the loss for that batch, updates the loss metric, and returns the result. The _compute_loss method calculates the loss (Triplet Loss). It subtracts the distance between the anchor and positive from the distance between the anchor and negative, and then adds a margin (0.5) to the result, ensuring that the value does not go below 0 by clipping it (setting the maximum value to 0). This resulting value is the triplet loss.
  <p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/c72eef41-5207-486f-bb8f-1846050ac12e" />
</p>
 Finally, I create an instance of the SiameseModel class, compile the model using the Adam optimizer with a learning rate of 0.0001, and proceed with training for 10 epochs.
  <p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/d279a95c-df2a-482b-9cc2-e9d7e608d749" />
</p>

## 2.5 image retrieval.
I load the previously trained model in order to get features of database.At this point, I perform feature extraction for all image files present in the specified folder and its subfolders. I achieve this through the "get_bottom_folders" function, which recursively retrieves the deepest level of folders in the specified directory. Then, within a for loop, I use the predict method to extract features and obtain file paths for each image file in the subfolders.
  <p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/d248392b-9eac-4a60-8c0e-025de6deeea7" />
</p>
  <p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/ea926007-a29e-45d4-876b-d5382986a8ee" />
</p>

The same process is used to obtain the features of the query image. 
</p>
  <p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/55ea0244-4162-4ad5-ab0c-1c5d0c059b09" />
</p>
</p>
  <p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/0c91c31c-2ab4-454f-9d3d-0f5ebcdb9084" />
</p>

Once I have obtained the feature vectors for both the database and the query image, the next step is to calculate the similarity between the feature vectors of the query image and the database images to detect highly similar images. This process is carried out using three functions: "def cos_sim," "def get_top_n_indexes," and "def search". These functions take the query vector, the list of feature vectors (features), and the list of file names (file_names) as input and calculate the Cosine similarity. The results are sorted in descending order by similarity value, and the corresponding indexes, similarity values, and file names are returned.
</p>
  <p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/1659dcb2-59af-4822-beb5-f260a382bd43" />
</p>

To execute the similarity calculation for a specific query image name, I call the "def search" function and specify the query image name.
</p>
  <p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/8082bd41-91d2-4bcd-9fa1-82230b9332ce" />
</p>

Finally, the function outputs the top 20 images with the highest similarity.
</p>
  <p align="center">
  <img src="https://github.com/makoto0825/image_retrieval/assets/120376737/fb9373ef-f203-482c-a8ea-e83212610bb5" />
</p>
