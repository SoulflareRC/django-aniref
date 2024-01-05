import { useState, useEffect } from 'react'
import Card from '@mui/material/Card';
import { CardHeader, CardActions, Box, Typography, Grid } from '@mui/material';
import CardContent from '@mui/material/CardContent';
import AccountCircle from '@mui/icons-material/AccountCircle';
import Image from 'next/image';
import { ENDPOINTS } from '@/endpoints';

import {List,ListItem,ListItemText,ListItemIcon,LinearProgress,LinearProgressWithLabel} from '@mui/material';
import {TASK_STATE, state_progress } from '@/constants';
import {  TaskCardLayout } from '@/components/taskCardLayout';
import { redirect } from 'next/dist/server/api-utils';
import { useRouter } from 'next/router';

const Task = (props) =>{
    // const {data} = props; 

    const router = useRouter(); 
    const {id,img,date} = props; 
    const [val,setVal] = useState(null);
    const [state,setState] = useState(props.state); 
    const flag = false; 
    const ws_url = ENDPOINTS.WS.ANIREF(`${id}/`) // `ws://localhost:8000/ws/app/tasks/aniref/${id}/`;
    console.log(state); 
    useEffect(()=>{
      const socket = new WebSocket(ws_url);
      socket.onmessage=(msg)=>{
        let data = JSON.parse(msg.data);
        // console.log("Received message for task id:",id);
        console.log("Received message:",data);  
        setState(data.info.message);
        if(data.info?.status=="SUCCESS"){
          console.log(router.asPath,router.pathname)
          router.push(`${router.asPath}/result/`)
        }else if(data.info?.status=="FAILURE"){
          console.log("Task failed"); 
        }
      }; 
      return ()=>{
        if(socket.readyState===1){
          socket.close();
        }
      }
    },[]); 
  return (
    <TaskCardLayout {...props}>
        <Box border={0}>
          <Grid container columns={12}>
            <Grid item xs={1} display={"flex"} alignItems={"center"}>
              <Typography fontSize={"small"} textAlign={'center'} color={"text.secondary"}>Task status:</Typography>
            </Grid>
            <Grid item xs={9}  border={0} display={"flex"} alignItems={"center"} justifyContent={"center"}>
              <Box width={"100%"} px={2} border={0}>
                <LinearProgress sx={{borderRadius:3}}  />
              </Box>
            </Grid>
            <Grid item xs={2} display={"flex"}  alignItems={"center"}>
              <Box width={"100%"}>
                <Typography width={"100%"} fontSize={"small"} textAlign={'center'} color={"text.secondary"}>{state}</Typography>
              </Box>
            </Grid>
          </Grid>
        </Box>
        <Box flexGrow={0}  width={"auto"} maxHeight={"auto"} border={0} py={1} display={"flex"} justifyContent={"center"} alignItems={"center"}>
          <Image priority src={img}  alt={"No Ref Image"} width={0} height={0} sizes='35em' 
          style={{borderWidth:0,borderColor:"black",borderStyle:"solid",borderRadius:3, width:"auto",height:"auto"}}/>
        </Box>
    </TaskCardLayout>
  );
};
export default Task;


export async function getServerSideProps(context) {
    const {resolvedUrl} = context; 
    const task_id = context.params.id; 
    const response = await fetch(ENDPOINTS.TASKS.ANIREF(`?task_id=${task_id}`,true)); 
    const data = (await response.json())[0];
    console.log(data);  
    if(!data) return {props:{}}
    const task_state = data.task_result.status; 
    // console.log(`Status of task ${task_id}:${task_state}`); 
    // console.log(resolvedUrl); 
    if(task_state===TASK_STATE.SUCCESS){
        // redirect to result page 
        return {
            redirect:{
                permanent:false, 
                destination: `${resolvedUrl}/result`, 
            }, 
            props:{} 
        }
    }

    const props = {
      id: task_id, 
      state: task_state,
      img: data.ref_img, 
      name: data.name, 
      date: data.task_result.date_created, 
    }  ;
    // Pass data to the page via props
    return { props }
  }
