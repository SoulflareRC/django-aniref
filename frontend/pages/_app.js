// import '@/styles/globals.css'
// import 'bootstrap/dist/css/bootstrap.css' 
import '@/styles/styles.css'
import "yet-another-react-lightbox/styles.css";
import {SessionProvider} from "next-auth/react";
import { Layout  } from "@/components/layout";
import {registerPlugin} from 'react-filepond'; 

import FilePondPluginMediaPreview from "filepond-plugin-media-preview";

import FilePondPluginImagePreview from 'filepond-plugin-image-preview'; 
import FilePondPluginFileValidateType from 'filepond-plugin-file-validate-type'; 
import 'filepond/dist/filepond.css'; 
import 'filepond-plugin-image-preview/dist/filepond-plugin-image-preview.min.css'; 
import 'filepond-plugin-media-preview/dist/filepond-plugin-media-preview.css'; 
registerPlugin(FilePondPluginMediaPreview,FilePondPluginImagePreview,FilePondPluginFileValidateType); 


export default function App({Component, pageProps: {session, ...pageProps}}) {
  return (
    <SessionProvider session={session}>
      <Layout>
        <Component {...pageProps} />
      </Layout>
    </SessionProvider>
  );
}
