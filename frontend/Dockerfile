FROM node:20-alpine

WORKDIR /
RUN mkdir frontend
WORKDIR /frontend 
ADD . /frontend
RUN npm install --force
CMD ["npx","next","dev"]