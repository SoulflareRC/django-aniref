import { useState } from "react";
import { FilePond } from "react-filepond";
export const FilepondInput = (props) => {
    const [file,setFile] = useState(null); 
    return (
        <FilePond files={file}  onupdatefiles={setFile} storeAsFile {...props}/>
    )
}