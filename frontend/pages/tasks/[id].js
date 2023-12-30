import Head from 'next/head'; 
import { useState } from 'react';
import { useEffect } from 'react';
const Task = (props) =>{
  // const {data} = props; 
  const {id} = props; 
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
  return (
    <div> 
      <p>Task ID:{id}</p>
      {state!="SUCCESS" && 
      <p>Progress: {val}</p>} 
      {state=="SUCCESS"&&
      <p> Task Completed!!!</p>
      }
    </div>
  )
}
export default Task; 
export async function getServerSideProps(context) {
    const task_id = context.params.id; 
    // Fetch data from external API
    // const res = await fetch(`https://.../data`)
    // const data = await res.json()
    // get task state first, if yes then no further rendering 
    const task_state_url = "http://localhost:8000/app/api/task/state/"
    const task_type = "progress_task"  
    const response = await fetch(task_state_url+`?task_id=${task_id}&task_type=${task_type}`);
    const data =await response.json();
    console.log(data);  
    const task_state = data.state; 
    console.log(`Status of task ${task_id}:${task_state}`); 
    const props = {
      id: task_id, 
      state: task_state,
    }  ;
    // Pass data to the page via props
    return { props }
  }