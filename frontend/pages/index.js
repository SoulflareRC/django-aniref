import Head from 'next/head'
import Image from 'next/image'
import { Inter } from 'next/font/google'
import styles from '@/styles/Home.module.css'

// import { io } from 'socket.io-client'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import {signIn, useSession} from "next-auth/react"
import { Button, Grid } from '@mui/material'
// import { MuiFileInput } from "mui-file-input"
import axios from 'axios'

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';




// const socket = io('ws://localhost:8000/ws/app/progress/');
const ws_url =  "ws://localhost:8000/ws/app/progress/"; 
const Index = () => {
  const [taskCompleted, setTaskCompleted] = useState(false);
  const [ msg,setMsg] = useState("asdas");
  const router = useRouter();  

  const {data: session, status} = useSession();
  console.log(useSession()); 

  useEffect(() => {
    const checkTaskStatus = async () => { 
      try {
          const socket = new WebSocket(ws_url);

          socket.onopen = () => {
            console.log('Connected to WebSocket');
          };

          // socket.on('disconnect', () => {
          //   console.log('Disconnected from WebSocket');
          // });
          socket.onmessage = msg =>{
            let data = JSON.parse(msg.data);
            // data = data.data 
            data= data.message; 
            console.log('Received message:', data);
            setMsg(data.progress); 
          }

          // Clean up: disconnect the socket when component unmounts
          return () => {
            socket.close(); 
            socket.onmessage=null; 
            socket.onopen=null; 
          };
        // }
      } catch (error) {
        console.error('Error fetching task status:', error);
      }
    };

    checkTaskStatus();
  }, []);
  const startTask = async () => {
    const taskURL = 'http://localhost:8000/app/api/progress/'; 
    const response = await axios.post(taskURL,{});
    const data = response.data; 
    console.log(data); 
    const taskID = data.task_id; 
    console.log(taskID);  
    router.push(`/tasks/${taskID}`); 
  }
  // if (session) {
  //   router.push("profile");
  //   return;
  // }
  const [file,setFile] = useState(null); 
  return (
       <Card sx={{ width:"90%",height:"80%" }} elevation={5}>
        <CardContent sx={{height:"auto"}}>
        {/* {msg}
      <div className='d-flex gap-2'>
       <Button onClick={startTask}>Start a task!</Button> 
       <Button onClick={()=>{signIn(undefined,"/profile")}}>Sign in</Button>

       </div> */}
       <Grid container spacing={0} columns={12} sx={{height:"100%",border:1}}>
        <Grid item xs={2} border={1} ></Grid>
       </Grid>
        </CardContent>
    </Card>
  );
};

export default Index;
