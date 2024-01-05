FROM node:20-alpine

# ENV NODE_ENV=production 
RUN echo $NODE_ENV
WORKDIR /
RUN mkdir frontend
WORKDIR /frontend 
ADD . /frontend
RUN npm install --force
RUN NODE_ENV=production npm run build 
# CMD ["npx","next","dev"]
CMD ["sh","-c","NODE_ENV=production npm start"] 