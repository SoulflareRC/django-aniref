const WS_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_WS_URL;
const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL; 
// const BASE_URL_CLIENT = "http://127.0.0.1:8000/"; 
// const WS_BASE_URL_CLIENT = "ws://127.0.0.1:8000/";
// const API_BASE_URL = `${BASE_URL}app/api/` 
// ${process.env.NEXT_PUBLIC_BACKEND_URL}app/api/tasks/aniref-tasks/
export const ENDPOINTS = {
    TASKS:{
        ANIREF(task_id="",client=true){
            // return (client?BASE_URL_CLIENT:BASE_URL)+"app/api/tasks/aniref-tasks/"+task_id; 
            return BASE_URL+"app/api/tasks/aniref-tasks/"+task_id; 
        }
    }, 
    AUTH: {
        SIGNUP: BASE_URL+"auth/"+"register/" 
    }, 
    WS:{
        ANIREF(task_id="",client=true){
            // return (client?WS_BASE_URL_CLIENT:WS_BASE_URL)+`ws/app/tasks/aniref/${task_id}`; 
            
            return WS_BASE_URL+`ws/app/tasks/aniref/${task_id}`;
        }
    }
}