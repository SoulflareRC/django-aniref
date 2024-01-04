# django-aniref 
![img](/demo/result.png) 
This project builds a machine learning app capable of handling high volumes of users. The machine learning part of this project is based on my previous project [Aniref2-yolov8](https://github.com/SoulflareRC/Aniref2-yolov8/tree/main).
### What does AniRef do?
**Given a reference image of a character and a video, it automatically extracts images of the character from the video.** <br> <br>
Finding good reference images has always been a challenging and time consuming task for artists. Most of the times it's tedious and hard to manually get reference images from the anime outselves and we end up relying on the artbooks published by the anime studios. This project mainly presents a toolchain for artists to quickly extract reference images of their desired characters from anime videos.  We first use an object detection model to crop out the characters, and then use [CLIP](https://github.com/openai/CLIP) to extract features of target image and extracted character crops, and finally find images of the target character by computing cosine similarity between target vector and character crop feature vectors.  
### What does this project have on top of Aniref2? 
This project presents a production-ready system with a backend built with Django and a frontend built with Next.js+MUI that can provide machine learning service to a large amount of users. Because of the computation-intensive nature of machine learning tasks, this project uses Celery 
as a distributed task queue to process machine learning tasks asynchronously, uses Redis as the message broker, and uses Websocket to update the progress of the task in real-time with the frontend. This project also implements a user & authentication system supporting both the traditional username+password credential login and the convenient Google account login with dj-rest-auth and NextAuth.js for users to manage their tasks' results. 
### Pages Demo 
#### Main 
![img](/demo/main.png) 
#### In Progress
![img](/demo/task.png) 
#### Task Results 
![img](/demo/result2.png)
#### Sign In 
![img](/demo/signin.png) 
#### Profile 
![img](/demo/profile.png)
#### Installation 
This project uses Docker for quick setup. To launch the backend, run 
```
docker compose up
```
in the root directory. To launch the frontend, run 
```
npm install
npx next dev 
```
Then visit `http://127.0.0.1:3000` for the app. 
