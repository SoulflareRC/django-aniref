import JSZip from "jszip";
import {saveAs} from "file-saver"; 

export const downloadAllFiles = async (links) => {
    const zip = new JSZip(); 
    await Promise.all(links.map(async (link,idx)=>{
        const fname = getFname(link); 
        // console.log(link,fname); 
        const blob = await fetch(link).then((res)=>res.blob()); 
        // console.log(blob); 
        zip.file(fname,blob); 
    })); 
    const zipBlob = await zip.generateAsync({type:"blob"}); 
    // console.log(zipBlob); 
    saveAs(zipBlob,"result.zip"); 
}
const getFname = (url) => {
    const parts = url.split('/'); 
    return parts[parts.length-1]; 
}
