import { useState, useEffect, useCallback } from 'react'
import Card from '@mui/material/Card';
import { CardHeader, CardActions, CardMedia, Box, Typography, Grid, CardActionArea } from '@mui/material';
import CardContent from '@mui/material/CardContent';
import AccountCircle from '@mui/icons-material/AccountCircle';
import Image from 'next/image';
import { ENDPOINTS } from '@/endpoints';
import Lightbox from 'yet-another-react-lightbox';
import FileDownloadIcon from "@mui/icons-material/FileDownload"

import {Button,IconButton, List,ListItem,ListItemText,ListItemIcon,LinearProgress,LinearProgressWithLabel} from '@mui/material';
import {TASK_STATE, state_progress } from '@/constants';
import {  TaskCardLayout } from '@/components/taskCardLayout';
import { redirect } from 'next/dist/server/api-utils';
import { downloadAllFiles } from '@/lib/files';
import { useSession, getSession } from 'next-auth/react';

const Task = (props) =>{
    // const {data} = props; 
    const {id,img,video,date,results} = props; 
    const [val,setVal] = useState(null);
    const [state,setState] = useState(props.state); 
    const ws_url = "ws://localhost:8000/ws/app/progress/";
    console.log(state); 
    useEffect(()=>{
      
      const socket = new WebSocket(ws_url);
      socket.onmessage=(msg)=>{
        let data = JSON.parse(msg.data);
        data= data.message; 
        const taskID = data.task_id; 
        console.log("Received message for task id:",taskID); 
        if(taskID==id){ 
          setVal(data.progress);
        } 
      }; 
      return ()=>{
        socket.close();
      }
    },[]); 

    const [idx,setIdx] = useState(-1); 
    const photos = results.map(item=>{
        return {src:item.img}; 
    })
  return (
    <TaskCardLayout {...props}>
        <Grid container columns={12} columnSpacing={2} height={"100%"}>  
            <Grid item border={0} xs={12} md={4}>
                <Card  elevation={0} sx={{mb:3,border:0}} height="100%">
                    <CardHeader sx={{border:0,p:0.5}} title={<Typography fontWeight="bold" color={"text.secondary"}>Reference Image</Typography>}/>
                    <CardMedia title={"Reference Image"} sx={{p:1}}>
                        <Box flexGrow={1}  width={"100%"} border={0} display={"flex"} justifyContent={"center"} alignItems={"center"}>
                            <Image priority src={img}  alt={"No Ref Image"} width={0} height={0} sizes='35em' 
                            style={{borderWidth:0,borderColor:"black",borderStyle:"solid",borderRadius:3,  
                                    width:"100%",height:"auto"}}/>
                        </Box>
                    </CardMedia>
                </Card>
                
                <Card elevation={0} sx={{border:0}} height="100%">
                    <CardHeader sx={{border:0,p:0.5}} title={<Typography fontWeight="bold" color={"text.secondary"}>Reference Image</Typography>}/>
                    <CardMedia title={"Reference Image"} sx={{p:1}}>
                        <Box overflow={"hidden"} flexGrow={1}  width={"100%"} border={0} display={"flex"} justifyContent={"center"} alignItems={"center"}>
                            <video priority loop controls style={{ borderRadius:3, width: '100%', height: 'auto' }}>
                                <source src={video} />
                            </video>
                        </Box>
                    </CardMedia>
                </Card>
                
            </Grid>            
            <Grid item border={0} xs={12} md={8}>
                <Card  elevation={0} sx={{mb:3,border:0}} height="100%">
                    {/* <CardHeader sx={{border:0,p:0.5}} title={<Typography fontWeight="bold" color={"text.secondary"}>Results</Typography>}/> */}
                    <CardHeader
                        sx={{
                        border: 0,
                        p: 0.5,
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        }}
                        title={
                        <Typography fontWeight="bold" color="text.secondary">
                            Results
                        </Typography>
                        }
                        action={
                        <Button onClick={()=>downloadAllFiles(results.map(item=>item.img))} startIcon={<FileDownloadIcon/>} color="primary">
                            Download
                        </Button>
                        }
                    />
                    <CardContent>
                        <Grid columns={12} columnSpacing={2} container border={0}> 
                                    
                            {results.map((item,idx)=>{
                                return (
                                    <Grid key={idx} xs={3} p={1} borderRadius={3} border={0} sx={{height:"100px",overflow:"hidden",display:"flex",alignItems:"center",justifyContent:"center"}} onClick={()=>setIdx(idx)}>
                                        <Box height={"100%"} width={"auto"}  p={0} border={0}
                                        overflow={"hidden"}
                                        sx={{display:"flex",alignItems:"center",justifyContent:"center"}}
                                        >
                                            <Image priority src={item.img}  alt={"No Ref Image"} width={0} height={0} sizes='35em' 
                                            style={{borderRadius:5,  
                                            width:"auto",height:"100%",
                                            padding:2,maxHeight:"100px"}}/>
                                        </Box>
                                    </Grid> 
                                ); 
                            })}
                            <Lightbox 
                            index={idx}
                            slides={photos}
                            open={idx>=0}
                            close={()=>setIdx(-1)}
                            />
                        </Grid>
                    </CardContent>
                </Card>
            </Grid>            
            
        </Grid>
    </TaskCardLayout>
  );
};
export default Task;
export async function getServerSideProps(context) {
    
    const task_id = context.params.id; 
    const response = await fetch(ENDPOINTS.TASKS.ANIREF(`?task_id=${task_id}`)); 
    const data = (await response.json())[0];  
    const {name,ref_img, video, results} = data; 
    const task_state = data.task_result.status;
    // console.log(results); 
    // console.log(`Status of task ${task_id}:${task_state}`); 
    const props = {
      id: task_id, 
      name: name, 
      state: task_state,
      img: ref_img, 
      video: video, 
      results: results, 
      date: data.task_result.date_created, 
    }  ;
    
    // Pass data to the page via props
    return { props }
  }
