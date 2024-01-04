const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL; 
const API_BASE_URL = `${BASE_URL}app/api/` 
// ${process.env.NEXT_PUBLIC_BACKEND_URL}app/api/tasks/aniref-tasks/
export const ENDPOINTS = {
    TASKS:{
        ANIREF(task_id=""){
            return API_BASE_URL+"tasks/"+"aniref-tasks/"+task_id; 
        }
    }, 
    AUTH: {
        SIGNUP: BASE_URL+"auth/"+"register/" 
    }
}