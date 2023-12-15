export const ENDPOINTS = {
    TASK:{
        WS_URL:"ws/app/progress/",
        STATE(task_id,task_type){
            return `app/api/task/state/?task_id=${task_id}&task_type=${task_type}`; 
        }

    }
}