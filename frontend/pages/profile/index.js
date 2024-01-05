// pages/profile.tsx

import {useState} from "react";
import {signOut, useSession,getSession, signIn} from "next-auth/react";
// import {Box, Button, Code, HStack, Spinner, Text, VStack} from "@chakra-ui/react";
import {Card,CardActions, CardActionArea, CardHeader,CardContent,Grid,Box,Typography,Button,IconButton,Dialog,DialogActions,DialogTitle, DialogContent, DialogContentText} from '@mui/material'; 
import Image from "next/image";
import ListAltIcon from "@mui/icons-material/ListAlt" 
import { DeleteOutline } from "@mui/icons-material";
import { Add } from "@mui/icons-material";
import axios from "axios";
import { ENDPOINTS } from "@/endpoints";
import { useRouter } from "next/router";


export default function Home(props) {
  let {tasks} = props;
  console.log("Tasks:",tasks)
//   tasks = Array(18).fill(tasks[0]); 
  const router = useRouter(); 
  const {data: session, status} = useSession({required: true});
  const [response, setResponse] = useState("{}");
  const [delTask,setDelTask] = useState(null); 
  const handleDelete = (e,task) => {
    // Handle the delete action here
    e.preventDefault(); 
    e.stopPropagation(); 
    setDelTask(task); // Open the confirmation dialog
  };

  const handleDelTask = async (pk) => {
    const delURL = ENDPOINTS.TASKS.ANIREF(`${pk}/`)
    console.log("Del URL:",delURL); 
    const response = await axios.delete(delURL); 
    setDelTask(null); 
    router.reload(); 
  } 

  console.log(session); 

  if (status == "loading") {
    return <div>Loading...</div>
  }

  if (session) {
    return (
        <Card sx={{p:2, display:"flex", flexDirection:"column",  borderRadius: 5, width: "90%", height: "90%", overflow: "auto" }} elevation={5}>
        <CardHeader  avatar={
            <ListAltIcon/>
        } title={"My Tasks"}
        action={<Button href="/" startIcon={<Add/>}>New Task</Button>}
        />
        <CardContent sx={{display:"flex",flexDirection:"column",  flexGrow:1, height: "auto" }}>
            <Grid container columns={12} border={0} spacing={2} columnSpacing={0} height={"auto"}>
                {tasks.map((task,idx)=>{
                    console.log(task); 
                    if(!task)return null; 
                    const {task_id,ref_img,name,status,date} = task; 
                    return (
                        <Grid key={idx} item xs={3} height={"auto"} border={0}>
      <CardActionArea href={`/tasks/aniref/${task_id}`} sx={{border:0,flexGrow:1, position: "relative", display:"flex", flexDirection:"column",height:"100%",width:"100%" }}>
  <Card
    sx={{
      border: 0,
      p: 0,
      display: "flex",
      flexDirection: "column",
      borderRadius: 2,
      height: "100%",
      width: "100%",
      overflow: "auto",
    }}
    elevation={2}
  >

      <CardHeader
        sx={{ py: 1, px: 1, color: "text.secondary" }}
        title={<Typography fontSize={12} textAlign={"left"}>{name ? name : "Untitled Task"}</Typography>}
        subheader={<Typography fontSize={10} textAlign={"left"}>{new Date(date).toLocaleDateString()}</Typography>}
        action={
          <IconButton onClick={(e) => handleDelete(e, task)} aria-label="delete">
            <DeleteOutline fontSize="small" color="error" />
          </IconButton>
        }
      />

      <CardContent
        sx={{
          border: 0,
          p: 0,
          display: "flex",
          flexDirection: "column",
          flexGrow: 1,
          height: "auto",
        }}
      >
        <Box
          flexGrow={1}
          display="flex"
          justifyContent="center"
          alignItems="center"
          border={0}
          width={"100%"}
        >
          <Image
            priority
            src={ref_img}
            alt={"No Ref Image"}
            width={0}
            height={0}
            sizes="35em"
            style={{
              borderWidth: 0,
              borderColor: "black",
              borderStyle: "solid",
              borderRadius: 0,
              width: "auto",
              maxWidth: "100%",
              height: "auto",
              maxHeight: "200px",
            }}
          />
        </Box>
      </CardContent>

    <CardActions sx={{ border: 0, marginTop: "auto" }}>
      <Typography width={"100%"} border={0} textAlign={"right"} fontSize={8} color={"text.secondary"}>
        {status}
      </Typography>
    </CardActions>


  </Card>
  </CardActionArea>
</Grid>

                    )
                })}
            </Grid>
            {/* Dialog */}
            <Dialog open={delTask} onClose={()=>setDelTask(null)}>
                <DialogTitle>Confirm Delete</DialogTitle>
                {/* Add content and actions for confirmation dialog */}
                {/* For example: */}
                <DialogContent>
                    <DialogContentText>
                    Are you sure you want to delete this task?(Task Name:{delTask?.name} ID:{delTask?.task_id})
                    </DialogContentText>
                </DialogContent>
                <DialogActions>
                    <Button onClick={()=>setDelTaskID(null)}>Cancel</Button>
                    <Button variant="contained" color="error" onClick={()=>handleDelTask(delTask.pk)}>Delete</Button>
                </DialogActions>
            </Dialog>


        </CardContent>
      </Card>
    );
  }

  return <></>;
}

export async function getServerSideProps(context) {
    const session = await getSession(context); 
    console.log("Session in serverside:",session); 
    if (!session) {
        return {
            redirect:{
                permanent:false, 
                destination: `/api/auth/signin`, 
            }, 
            props:{} 
        }
    }else{
        const user_id = session.user.pk; 
        console.log(user_id); 
        const response = await axios.get(ENDPOINTS.TASKS.ANIREF(`?user=${user_id}`,false));
        const data = await response.data;  
        console.log("Response data:", data); 
        const tasks = data.map(item=>{
            return {
                pk: item.id, 
                date: item.task_result.date_created,
                task_id: item.task_id, 
                ref_img: item.ref_img, 
                name: item.name, 
                status: item.task_result.status, 
            }
        }); 
        // Pass data to the page via props
        const props = {
            tasks:tasks?tasks:[]
        }
        return { props }
    }
  }