# Aspect-Based Sentiment Analysis 

## The Problem
social media / Hotel managment websites like TripAdvisor, facebook, and Booking.com are a vast and valuable, yet untapped resource of feedback that hotels can use to improve their ratings and attract more customers. However, hotel staff do not have time to trawl through hundreds of written reviews to gather all the areas of improvement, and even numerical ratings are insufficient, as they cannot tell the hotel _how_ to improve. If there is a poor service rating, is it due to the front-counter experience, the room service or the cleaning service?

## How this project solves it
Aspect-Based Sentiment Analysis (ABSA) identifies aspects and their corresponding perception (positive, neutral, negative) within a sentence. For example, the review "the concierge staff were slow and rude to me" would yield the aspect "concierge" with sentiment "negative". This is a perfect method to use on unstructured hotel review data. By using an averaged scoring system on the most common aspects that appear in the reviews, not only can hotel staff zoom in on the exact issue bringing their ratings down, but also know more about the common aspects that their guests take note of in their feedback, both positive and negative. From the previous example, if "concierge" has the most complains, the staff now know their poor service rating is due to the counter staff, and focus on retraining those staff. The most benefits are achieved without wasting time on reading through every single feedback just to find the area of dissatisfaction.

## Pipeline
### Resources
- Model from [PyABSA](https://github.com/sumitsingh00/sentiment-analysis) repository -- Due to time constraints and limited labelled data, the model, which had already been trained on restaurant reviews, was further trained on 400 hotel reviews via transfer learning. 
- Labelled hotel review data from [SemEval 2015 Hotels Domain](http://metashare.ilsp.gr:8080/repository/browse/semeval-2015-absa-hotels-domain-test-data-gold-annotations/153796fc9ca211e4bf03842b2b6a04d73c1f9fdd8aff4c83884694f3ebf4e3b6/)
- Hotel reviews from [Kaggle](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe?resource=download) -- Used to extract out common aspects and contained target data for inference [Britannia Hotel]
- Britannia Hotel data used for inference, pretending that this hotel was our client who wanted to identify their top areas of improvement.
### Steps taken
1. Reformatted hotel reviews data into the accepted labelled format that the model took in for training data.
2. Reformatted model output to identify the frequencies of each aspect. The top 50 were chosen and manually sorted into regular hotel feedback categories as shown in the table below, with a few examples.

| Food          |   Service     |     Cost      |  Room Quality |  Amenities    |   Location    |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| food          | staff         |   price       |  bathroom     |     wifi      | convenient    |
| bar           | service       |   rates       |   clean       |    shower     |    near       |


3. Infer the results of Britannia reviews from the model, the aspects were sorted into the predefined categories in the table above and each category assigned a rating based on the formula: $$Average Score = \frac{[(positive freq) + negative freq(-1)]} {aspect freq}$$
4. Single out the lowest scoring category and within it, find the respective aspects with the highest frequency of negative sentiments presented. This will present the areas that needs the most improvement on.

## Usage of .ipynb file
- Access the ipynb file, and load the files from `/resources`. 
- The files in `/resources` are all the initial data used in the project; `/resources/output` are the files generated throughout the pipeline process as output for other functions or as final output (for example, the model checkpoints created after training, or the inference json results etc) 

## Hotel Britannia Results
Below is a summary of the results, with the top 50 aspects ranked according to frequency, then manually grouped into the table's categories. The 2 categories with the worst average scores are identified, and within the 2 worst categories, the aspects with the highest frequency of negative sentiment is found.

<img src="https://github.com/sumitsingh00/sentiment-analysis/blob/main/Images/fig1.jpg" width=80% height=80%>
</br>
</br>

In conclusion, room quality and food can be improved upon the most. Within room quality, that would mean improving the conditions of the shower, the air-conditioners and the wifi. For food, the breakfast could be better. The hotel staff can then focus on these areas by explicitly asking guests face-to-face what exactly contributes to the poor opinion of the respective categories.

## Conclusion
The project shows much promise given that it was done within 2 months with extremely limited training data, yet yielded rather decent results. With better cleaning techniques (perhaps grouping 'view' and 'views' together etc), and averaging system, a more accurate result can be reached in future iterations.

## Limitations and future improvements
-  The main constraints are due to the limited labelled data available. Data Augmentation can be used in future expansions to generate larger amounts of data for training.
- A better weighted formula could be used to more accurately determine the average of scores.
