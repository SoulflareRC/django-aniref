import Head from 'next/head'
export async function getServerSideProps(context) {
    const task_id = context.params.id; 
    // Fetch data from external API
    const res = await fetch(`https://.../data`)
    const data = await res.json()
   
    // Pass data to the page via props
    return { props: { data } }
  }