
import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import { signIn, useSession } from "next-auth/react"
import { Button, FormControl, Grid, TextField } from '@mui/material'
import axios from 'axios'

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import { FormLabel, Box, IconButton, Popover } from '@mui/material'
import { PopoverHint } from '@/components/popoverHint'
import { FilepondInput } from '@/components/filepondInput'

import { ENDPOINTS } from '../endpoints'
// const socket = io('ws://localhost:8000/ws/app/progress/');
// const ws_url = "ws://localhost:8000/ws/app/progress/";
const Index = () => {
  // const [taskCompleted, setTaskCompleted] = useState(false);
  // const [msg, setMsg] = useState("asdas");
  const router = useRouter();

  const { data: session, status } = useSession();
  console.log(useSession());
  const user = session?.user?.pk; 
  // useEffect(() => {
  //   const checkTaskStatus = async () => {
  //     try {
  //       const socket = new WebSocket(ws_url);

  //       socket.onopen = () => {
  //         console.log('Connected to WebSocket');
  //       };

  //       // socket.on('disconnect', () => {
  //       //   console.log('Disconnected from WebSocket');
  //       // });
  //       socket.onmessage = msg => {
  //         let data = JSON.parse(msg.data);
  //         // data = data.data 
  //         data = data.message;
  //         console.log('Received message:', data);
  //         setMsg(data.progress);
  //       }

  //       // Clean up: disconnect the socket when component unmounts
  //       return () => {
  //         socket.close();
  //         socket.onmessage = null;
  //         socket.onopen = null;
  //       };
  //       // }
  //     } catch (error) {
  //       console.error('Error fetching task status:', error);
  //     }
  //   };

  //   checkTaskStatus();
  // }, []);
  // const startTask = async () => {
  //   const taskURL = 'http://localhost:8000/app/api/progress/';
  //   const response = await axios.post(taskURL, {});
  //   const data = response.data;
  //   console.log(data);
  //   const taskID = data.task_id;
  //   console.log(taskID);
  //   router.push(`/tasks/${taskID}`);
  // }
  // if (session) {
  //   router.push("profile");
  //   return;
  // }
  const handleSubmit = async (e) => {
    e.preventDefault(); 
    let data = new FormData(e.target); 
    console.log(data.get("ref_img"),data.get("vid")); 
    const taskURL=ENDPOINTS.TASKS.ANIREF();  
    console.log(taskURL); 
    const response = await axios.post(taskURL,data); 
    console.log(response.data); 
    console.log(response.status); 
    const taskID = response.data.task_id; 
    router.push(`/tasks/aniref/${taskID}`);   
  }

  return (
    <Card sx={{ borderRadius: 5, width: "90%", height: "90%", overflow: "auto" }} elevation={5}>
      <CardContent sx={{ height: "auto" }}>
        {/* {msg}
      <div className='d-flex gap-2'>
       <Button onClick={startTask}>Start a task!</Button> 
       <Button onClick={()=>{signIn(undefined,"/profile")}}>Sign in</Button>

       </div> */}
        <form style={{ borderWidth: 0, borderColor: "black", borderStyle: "solid", height: "auto" }} onSubmit={handleSubmit} >
          {user && <input type='hidden' name='user' value={user}></input>} 
          <Grid container spacing={0} columns={12} sx={{ height: "100%", border: 0 }}>
            <Grid item xs={12} display={"flex"}>
              <FormControl sx={{border:0,p:2,width:"100%"}}>
                <TextField name='name' label="Name(Optional)" variant='standard'></TextField>
              </FormControl>
            </Grid>
            <Grid item xs={12} lg={4} border={0} height={"auto"} display={"flex"} justifyContent={"center"} alignItems={"top"}>
              <FormControl sx={{ p: 2, gap: 2, width: "100%" }}>
                <Box display={"flex"} alignItems={"center"} justifyContent={"space-between"}>
                  <FormLabel>Reference Image</FormLabel>
                  <PopoverHint m={2} hint={"The reference image."} />
                </Box>
                <Box border={0} height={"auto"} >
                  <FilepondInput required name="ref_img" acceptedFileTypes={"image/*"} />
                </Box>

              </FormControl>

            </Grid>

            <Grid item xs={12} lg={8} border={0} display={"flex"} justifyContent={"center"} alignItems={"top"}>
              <FormControl sx={{ p: 2, gap: 2, width: "100%" }}>
                <Box display={"flex"} alignItems={"center"} justifyContent={"space-between"}>
                  <FormLabel border={1} width={"100%"}>
                    Input Video
                  </FormLabel>
                  <PopoverHint m={2} hint={"The input video to extract corresponding images from."} />
                </Box>

                <Box border={0}>
                  <FilepondInput required name="video" acceptedFileTypes={"video/*"} />
                </Box>
              </FormControl>
            </Grid>
            <Grid item px={2} xs={12} border={0} display={"flex"}>
              <Button type='submit' variant='outlined' sx={{ flexGrow: 1 }}>
                Submit
              </Button>
            </Grid>
          </Grid>
        </form>
      </CardContent>
    </Card>
  );
};

export default Index;
