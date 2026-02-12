import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArrowDown, ArrowRight, ArrowUpRight, Check, CircleNotch, Clock, FilmStrip, FolderOpen, MagnifyingGlass, Play, Plus, ShieldCheck, SlidersHorizontal, Trash, UploadSimple, X, ChartBar, Info, Crosshair, Scissors, ThumbsUp, ThumbsDown } from '@phosphor-icons/react';
import '@fontsource-variable/geist';
import './style.css';

type Video = {id:string;filename:string;state:string;duration:number;error:string|null};
type Frame = {id:string;video_id:string;seconds:number;duration:number;filename:string;image_url:string;video_url:string;similarity:number;sharpness:number};
type Approval = {id:string;video_id:string;filename:string;start:number;end:number;state:string};
type Job = {id:string;kind:string;state:string;progress:number;error:string|null;result?:{download_url?:string}};
type Detection = {width:number;height:number;objects:{label:string;score:number;box:number[]}[]};
type Turn = {turn_id:string;query:string;answer:string;plan:string;matches:Frame[];approval:Approval|null;detection?:Detection;trace:{tool:string;status:string;milliseconds:number}[];seconds:number;budget:{used:number;maximum:number}};
type Metric = {accuracy:number;n:number;correct:number;mean_latency_seconds:number;estimated_dollars_per_correct:number;predictions:{id:string;expected:string;predicted:string;correct:boolean}[]};
type Evaluation = {executed_at?:string;task?:string;selected_stage?:string;split_sizes?:Record<string,number>;release_gate?:{passed:boolean;rule:string};metrics:Record<string,Metric>};

async function api<T>(path:string, options:RequestInit={}):Promise<T> {
  const controller=new AbortController();
  const timer=window.setTimeout(()=>controller.abort(),180000);
  try {
    const response=await fetch(path,{credentials:'same-origin',...options,signal:controller.signal});
    const body=await response.json();
    if(!response.ok) throw new Error(typeof body.detail==='string'?body.detail:'The request could not be completed. Check your inputs and retry.');
    return body;
  } catch(error) {
    if(error instanceof DOMException && error.name==='AbortError') throw new Error('The request timed out. The server may still be warming up; please retry.');
    throw error;
  } finally {window.clearTimeout(timer);}
}
const post=<T,>(path:string,data:unknown)=>api<T>(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
const timestamp=(value:number)=>`${Math.floor(value/60).toString().padStart(2,'0')}:${Math.floor(value%60).toString().padStart(2,'0')}`;
const message=(error:unknown)=>error instanceof Error?error.message:'Something went wrong. Please retry.';

function App(){
  const [tab,setTab]=useState('workspace'),[videos,setVideos]=useState<Video[]>([]),[jobs,setJobs]=useState<Job[]>([]);
  const [booted,setBooted]=useState(false),[busy,setBusy]=useState(''),[error,setError]=useState(''),[notice,setNotice]=useState('');
  const [query,setQuery]=useState(''),[turns,setTurns]=useState<Turn[]>([]),[matches,setMatches]=useState<Frame[]>([]);
  const [source,setSource]=useState(''),[sharpOnly,setSharpOnly]=useState(false),[selected,setSelected]=useState<Frame|null>(null);
  const [approval,setApproval]=useState<Approval|null>(null),[start,setStart]=useState(0),[end,setEnd]=useState(6);
  const [detection,setDetection]=useState<Detection|null>(null),[mask,setMask]=useState(''),[evaluation,setEvaluation]=useState<Evaluation|null>(null);
  const fileRef=useRef<HTMLInputElement>(null),videoRef=useRef<HTMLVideoElement>(null);
  const selectedRef=useRef(selected);selectedRef.current=selected;
  const jobsRef=useRef(jobs);jobsRef.current=jobs;
  const active=jobs.some(j=>j.state==='queued'||j.state==='running');
  async function refresh(){const result=await api<{videos:Video[]}>('/api/videos');setVideos(result.videos);}
  useEffect(()=>{api<{library:Video[]}>('/api/bootstrap').then(r=>{setVideos(r.library);setBooted(true);}).catch(e=>setError(message(e)));},[]);
  useEffect(()=>{
    if(!active)return;
    let stopped=false;
    const timer=window.setInterval(async()=>{
      try{
        const next=await Promise.all(jobsRef.current.map(j=>['queued','running'].includes(j.state)?api<Job>(`/api/jobs/${j.id}`):Promise.resolve(j)));
        if(stopped)return;
        setJobs(next);await refresh();
      }catch(e){if(!stopped)setError(message(e));}
    },2500);
    return()=>{stopped=true;window.clearInterval(timer);};
  },[active]);
  useEffect(()=>{if(tab==='evaluation')api<Evaluation>('/api/evaluation').then(setEvaluation).catch(e=>setError(message(e)));},[tab]);
  function addJob(job:Job){setJobs(old=>[job,...old].slice(0,12));}
  async function loadSample(){
    setBusy('sample');setError('');setNotice('');
    try{const result=await post<{job:Job}>('/api/sample',{});addJob(result.job);await refresh();setNotice('Sample added. Indexing runs in the background; the first run downloads the visual model.');}
    catch(e){setError(message(e));}finally{setBusy('');}
  }
  async function upload(file:File|undefined){
    if(!file)return;
    if(file.size>50*1024*1024){setError('Choose a video smaller than 50 MB.');return;}
    setBusy('upload');setError('');setNotice('');
    try{const result=await api<{job:Job}>('/api/videos',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Filename':encodeURIComponent(file.name)},body:file});addJob(result.job);await refresh();setNotice('Upload received. SceneTrace is preparing playable footage and visual evidence.');}
    catch(e){setError(message(e));}finally{setBusy('');if(fileRef.current)fileRef.current.value='';}
  }
  async function ask(value=query){
    if(!value.trim()||busy)return;
    setBusy('search');setError('');setNotice('');setQuery(value);
    try{
      const result=await post<Turn>('/api/agent',{query:value,frame_id:selected?.id||null,video_id:source||null,min_sharpness:sharpOnly?80:0});
      setTurns(old=>[result,...old].slice(0,8));
      if(result.plan==='search'){setMatches(result.matches);if(result.matches.length)choose(result.matches[0]);}
      if(result.approval)setApproval(result.approval);
      if(result.detection)setDetection(result.detection);
    }catch(e){setError(message(e));}finally{setBusy('');}
  }
  function choose(frame:Frame){
    setSelected(frame);setStart(Math.max(0,Math.round((frame.seconds-3)*10)/10));setEnd(Math.min(frame.duration,Math.round((frame.seconds+3)*10)/10));setDetection(null);setMask('');
    if(videoRef.current && selected?.video_id===frame.video_id)videoRef.current.currentTime=frame.seconds;
  }
  async function prepare(){
    if(!selected)return;
    setBusy('prepare');setError('');
    try{setApproval(await post<Approval>('/api/exports/prepare',{video_id:selected.video_id,start,end}));}
    catch(e){setError(message(e));}finally{setBusy('');}
  }
  async function approve(approved:boolean){
    if(!approval)return;
    setBusy('export');setError('');
    try{const result=await post<{job?:Job}>(`/api/exports/${approval.id}/approve`,{approved});if(result.job)addJob(result.job);setApproval(null);if(approved)setNotice('Approved. Your clip is rendering in the background.');}
    catch(e){setError(message(e));}finally{setBusy('');}
  }
  async function detect(){
    if(!selected)return;setBusy('detect');setError('');setMask('');
    try{setDetection(await post<Detection>('/api/frames/detect',{frame_id:selected.id}));}
    catch(e){setError(message(e));}finally{setBusy('');}
  }
  async function makeMask(box:number[]){
    if(!selected)return;setBusy('mask');setError('');
    try{const result=await post<{image_url:string}>('/api/frames/mask',{frame_id:selected.id,box});setMask(result.image_url+'?v='+Date.now());}
    catch(e){setError(message(e));}finally{setBusy('');}
  }
  async function remove(video:Video){
    if(!window.confirm(`Delete ${video.filename} and its exports from this workspace?`))return;
    try{await api(`/api/videos/${video.id}`,{method:'DELETE'});setMatches(old=>old.filter(f=>f.video_id!==video.id));if(selected?.video_id===video.id)setSelected(null);if(source===video.id)setSource('');setJobs(old=>old.filter(j=>j.state!=='ready'));await refresh();}
    catch(e){setError(message(e));}
  }
  async function clear(){
    if(!window.confirm('Delete all footage, exports and query history in this visitor workspace?'))return;
    try{await api('/api/workspace',{method:'DELETE'});setVideos([]);setMatches([]);setSelected(null);setTurns([]);setJobs([]);setApproval(null);setSource('');setQuery('');setNotice('Workspace cleared.');}
    catch(e){setError(message(e));}
  }
  async function feedback(turn:Turn,vote:number){
    try{await post('/api/feedback',{turn_id:turn.turn_id,vote,comment:''});setNotice('Feedback saved for this session. Thank you.');}
    catch(e){setError(message(e));}
  }
  const currentTurn=turns[0];
  return <div className="shell">
    <a className="skip" href="#main">Skip to workspace</a>
    <aside className="sidebar">
      <a href="/" className="brand" aria-label="SceneTrace home"><span className="brand-mark"><FilmStrip size={25} weight="duotone"/></span>SceneTrace<span className="beta">01</span></a>
      <div className="project-label">YOUR SCREENING ROOM<span>SESSION</span></div>
      <nav aria-label="Main navigation">
        <button className={tab==='workspace'?'nav active':'nav'} onClick={()=>setTab('workspace')}><FolderOpen size={19}/>Workspace<span>{videos.length.toString().padStart(2,'0')}</span></button>
        <button className={tab==='evaluation'?'nav active':'nav'} onClick={()=>setTab('evaluation')}><ChartBar size={19}/>Evaluation</button>
        <button className={tab==='about'?'nav active':'nav'} onClick={()=>setTab('about')}><Info size={19}/>How it works</button>
      </nav>
      <div className="library-heading">FOOTAGE <button aria-label="Upload a video" onClick={()=>fileRef.current?.click()} disabled={!booted||!!busy}><Plus size={18}/></button></div>
      <div className="library">
        {videos.length===0?<p className="empty-library">Your footage will appear here.<br/>Start with a sample or upload.</p>:videos.map(v=><div className={'library-item '+(source===v.id?'chosen':'')} key={v.id}>
          <button className="library-select" onClick={()=>{setSource(source===v.id?'':v.id);setTab('workspace');}}><FilmStrip size={18}/><span><strong>{v.filename}</strong><small>{v.state==='ready'?timestamp(v.duration):v.state==='failed'?'Processing failed':'Indexing footage…'}</small></span></button>
          <button className="delete" aria-label={`Delete ${v.filename}`} onClick={()=>remove(v)} disabled={v.state==='indexing'}><Trash size={15}/></button>
          {v.error&&<small className="file-error">{v.error}</small>}
        </div>)}
      </div>
      <div className="sidebar-bottom"><ShieldCheck size={20}/><strong>Yours for this session.</strong><p>Footage expires after 24 hours or a host restart. Download what you want to keep.</p><button className="text-button" onClick={clear} disabled={!booted||active}>Clear workspace <ArrowUpRight size={13}/></button></div>
    </aside>
    <div className="content">
      <header className="topbar"><div className="breadcrumb">SceneTrace <span>/</span> {tab==='workspace'?'Workspace':tab==='evaluation'?'Evaluation':'How it works'}</div><div className="top-actions"><span className="online"><i/>CPU workspace</span><a href="https://github.com/Eunini/scenetrace" target="_blank" rel="noreferrer">Source <ArrowUpRight size={13}/></a><button className="primary small" disabled={!booted||!!busy} onClick={()=>fileRef.current?.click()}><UploadSimple size={16}/>Upload video</button></div></header>
      <input ref={fileRef} type="file" accept="video/mp4,video/quicktime,video/webm,.mkv" className="sr-only" aria-label="Choose video file" onChange={e=>upload(e.target.files?.[0])}/>
      <main id="main">
        {error&&<div className="alert" role="alert"><span>{error}</span><button aria-label="Dismiss error" onClick={()=>setError('')}><X size={16}/></button></div>}
        {notice&&<div className="notice" role="status"><span>{notice}</span><button aria-label="Dismiss notification" onClick={()=>setNotice('')}><X size={16}/></button></div>}
        {tab==='workspace'&&<>
          <div className="workspace-heading"><div><div className="eyebrow">FROM FOOTAGE TO FINDINGS</div><h1>Find the shot.<br/><span>Keep the evidence.</span></h1></div><p>A search workspace for the moments<br className="desktop-break"/> hiding in your footage.</p></div>
          <form className="search-form" onSubmit={e=>{e.preventDefault();ask();}}><MagnifyingGlass size={22}/><label className="sr-only" htmlFor="query">Ask about your footage</label><input id="query" value={query} maxLength={500} onChange={e=>setQuery(e.target.value)} placeholder="Find a rabbit in the grass…" autoComplete="off"/><button className="primary" disabled={!booted||!!busy||!query.trim()} type="submit">{busy==='search'?<><CircleNotch className="spin" size={18}/>Searching</>:<>Search <ArrowRight size={18}/></>}</button></form>
          <div className="search-options"><div className="suggestions"><span>TRY</span>{['Find a rabbit','Show my videos','Total video duration'].map(q=><button key={q} disabled={!!busy||!booted} onClick={()=>ask(q)}>{q}<ArrowUpRight size={12}/></button>)}</div><div className="filters"><label><SlidersHorizontal size={15}/><select aria-label="Search source" value={source} onChange={e=>setSource(e.target.value)}><option value="">All footage</option>{videos.filter(v=>v.state==='ready').map(v=><option key={v.id} value={v.id}>{v.filename}</option>)}</select></label><label className="sharp-filter"><input type="checkbox" checked={sharpOnly} onChange={e=>setSharpOnly(e.target.checked)}/>Sharper frames</label></div></div>
          {jobs.length>0&&<section className="jobs" aria-label="Background jobs">{jobs.map(j=><div key={j.id} className={'job '+j.state}><span>{['queued','running'].includes(j.state)?<CircleNotch className="spin" size={15}/>:j.state==='ready'?<Check size={15}/>:<Info size={15}/>} {j.kind==='index'?'Video indexing':'Clip export'} · {j.state}</span>{['queued','running'].includes(j.state)&&<progress aria-label={`${j.kind} progress`} max={1} value={j.progress}/>} {j.error&&<span>{j.error}</span>}{j.result?.download_url&&<a className="download" href={j.result.download_url} download>Download MP4 <ArrowDown size={14}/></a>}</div>)}</section>}
          <div className="stage-grid">
            <section className="screening" aria-label="Video preview">
              {selected?<><div className="player"><video key={selected.video_id} ref={videoRef} src={selected.video_url} controls playsInline preload="metadata" onLoadedMetadata={()=>{if(videoRef.current)videoRef.current.currentTime=selectedRef.current?.seconds||0;}} aria-label="Selected source video"/></div><div className="player-caption"><span><i className="live-dot"/>{selected.filename}</span><span>{timestamp(selected.seconds)} / {timestamp(selected.duration)}</span></div><div className="clip-controls"><div className="clip-title"><Scissors size={17}/><strong>Make a clip</strong><span>Up to 30 seconds</span></div><div className="clip-fields"><label>Start (seconds)<input type="number" min={0} max={selected.duration} step={0.1} value={start} onChange={e=>setStart(Number(e.target.value))}/></label><span>—</span><label>End (seconds)<input type="number" min={0} max={selected.duration} step={0.1} value={end} onChange={e=>setEnd(Number(e.target.value))}/></label><button className="secondary" onClick={prepare} disabled={!!busy||!Number.isFinite(start)||!Number.isFinite(end)||end<=start||end-start>30||end>selected.duration}>Prepare export <ArrowRight size={15}/></button></div></div></>:<div className="empty-stage"><img src="/open-film-poster.jpg" alt="A frame from Big Buck Bunny, the included open-film sample"/><div className="empty-stage-shade"/><div className="sample-label">OPEN FILM / BLENDER FOUNDATION</div><div className="empty-stage-copy"><span className="round-play"><Play size={20} weight="fill"/></span><h2>Start with a story.<br/>Find your first scene.</h2><p>Explore an open-film sample, or bring your own footage.</p><button className="primary" onClick={loadSample} disabled={!booted||!!busy||active}>{busy==='sample'?<CircleNotch size={18} className="spin"/>:<Play size={17} weight="fill"/>}Explore sample <ArrowRight size={17}/></button><small>MP4, MOV or WebM · Up to 50 MB / 3 minutes</small></div><div className="sample-credit">Big Buck Bunny © 2008 Blender Foundation · CC BY 3.0</div></div>}
            </section>
            <aside className="activity-panel" aria-label="Search activity"><div className="panel-label"><span className="eyebrow">THE WORK BEHIND THE ANSWER</span><span className="activity-light"/></div><h2>Every result has a trail.</h2>{currentTurn?<><p className="answer" role="status">{currentTurn.answer}</p><div className="execution-summary"><span><Clock size={14}/>{currentTurn.seconds.toFixed(2)}s</span><span>{currentTurn.budget.used} / {currentTurn.budget.maximum} tools</span></div><ol className="trace">{currentTurn.trace.map((step,i)=><li key={i}><span className="trace-number">{i+1}</span><div><strong>{step.tool.replaceAll('_',' ')}</strong><small>{step.status} · {step.milliseconds}ms</small></div><Check size={14}/></li>)}</ol><div className="feedback"><span>Was this useful?</span><button aria-label="Search was useful" onClick={()=>feedback(currentTurn,1)}><ThumbsUp size={17}/></button><button aria-label="Search was not useful" onClick={()=>feedback(currentTurn,-1)}><ThumbsDown size={17}/></button></div></>:<><p>Ask a question. SceneTrace chooses a bounded set of tools and shows the source behind each match.</p><ol className="intro-steps"><li><span>01</span><div><strong>Add your footage</strong><p>We sample and index visual frames.</p></div></li><li><span>02</span><div><strong>Ask for a moment</strong><p>Search scenes in your own words.</p></div></li><li><span>03</span><div><strong>Check, then export</strong><p>Play the source. Approve your clip.</p></div></li></ol></>}<div className="evidence-note"><ShieldCheck size={17}/><p>Visual similarity suggests candidates. Playback is your source of truth.</p></div></aside>
          </div>
          {approval&&<section className="approval" aria-label="Export approval"><div><div className="eyebrow">YOUR APPROVAL IS REQUIRED</div><h2>Export {timestamp(approval.start)}–{timestamp(approval.end)}?</h2><p>{approval.filename} · {(approval.end-approval.start).toFixed(1)} seconds · MP4</p></div><div><button className="secondary" onClick={()=>approve(false)} disabled={!!busy}>Cancel</button><button className="primary" onClick={()=>approve(true)} disabled={!!busy}>Approve and export <Check size={17}/></button></div></section>}
          <section className="results" aria-label="Visual search results"><div className="section-heading"><h2>Visual evidence <span>{matches.length.toString().padStart(2,'0')}</span></h2><p>Sampled frames · Ranked by visual similarity</p></div>{matches.length?<div className="results-grid">{matches.map((frame,i)=><button className={'frame-card '+(selected?.id===frame.id?'selected':'')} key={frame.id} onClick={()=>choose(frame)} aria-label={`Play result ${i+1} at ${timestamp(frame.seconds)}`}><div className="frame-image"><img src={frame.image_url} alt={`Sampled frame at ${timestamp(frame.seconds)} from ${frame.filename}`} loading="lazy"/><span className="frame-play"><Play size={16} weight="fill"/></span><span className="timecode">{timestamp(frame.seconds)}</span></div><div className="frame-info"><strong>{frame.filename}</strong><div><span>#{(i+1).toString().padStart(2,'0')} match</span><span>{frame.similarity.toFixed(3)} similarity</span></div></div></button>)}</div>:<div className="empty-results"><FilmStrip size={27}/><p>Your search results will appear here.<span>Every match opens the original footage at its timestamp.</span></p></div>}</section>
          {selected&&<section className="inspector"><div className="section-heading"><div><h2>Inspect a frame</h2><p>Optional object detection and a bounding-box-assisted mask.</p></div><button className="secondary" onClick={detect} disabled={!!busy}><Crosshair size={17}/>{busy==='detect'?'Detecting…':'Detect objects'}</button></div>{detection&&<div className="detection-grid"><div className="detection-image"><img src={selected.image_url} alt="Selected frame with predicted object boxes"/>{detection.objects.map((o,i)=><span key={i} className="bbox" style={{left:`${o.box[0]/detection.width*100}%`,top:`${o.box[1]/detection.height*100}%`,width:`${(o.box[2]-o.box[0])/detection.width*100}%`,height:`${(o.box[3]-o.box[1])/detection.height*100}%`}}><b>{o.label}</b></span>)}</div><div><h3>{detection.objects.length} detected objects</h3>{detection.objects.length===0&&<p>No COCO-category objects passed the detector’s threshold. This does not prove the frame is empty.</p>}{detection.objects.map((o,i)=><div className="object-row" key={i}><span>{o.label} <small>{(o.score*100).toFixed(0)}% detector score</small></span><button className="secondary small" disabled={!!busy} onClick={()=>makeMask(o.box)}>Make mask</button></div>)}<p className="muted">DETR boxes are predictions. Masks use classical GrabCut, not a semantic segmentation model.</p>{mask&&<a href={mask} download="scenetrace-mask.png"><img className="mask" src={mask} alt="Estimated foreground mask"/>Download PNG <ArrowDown size={14}/></a>}</div></div>}</section>}
          {turns.length>1&&<details className="history"><summary>Earlier requests in this session <span>{turns.length-1}</span></summary>{turns.slice(1).map(turn=><div key={turn.turn_id}><strong>{turn.query}</strong><p>{turn.answer}</p><small>{turn.plan} · {turn.seconds.toFixed(2)}s · {turn.budget.used} tools</small></div>)}</details>}
        </>}
        {tab==='evaluation'&&<section className="evaluation"><div className="eyebrow">MEASURE BEFORE SHIPPING</div><h1>A small model.<br/><span>An inspectable experiment.</span></h1><p className="lede">Actual SFT and group-relative policy optimization on an open-weight language model. This benchmark measures five-way tool routing, not general video understanding.</p>{evaluation&&<><div className="evaluation-status"><ShieldCheck size={22}/><div><strong>{evaluation.release_gate?.passed?'Release gate passed':'Evaluation in progress'}</strong><p>Serving the {evaluation.selected_stage?.toUpperCase()||'evaluated'} adapter selected on validation questions.</p></div></div><div className="metrics-grid">{['base','sft','grpo'].map(stage=>{const m=evaluation.metrics[`${stage}_test`];return <article key={stage}><span className="eyebrow">{stage==='base'?'BASE MODEL':stage==='sft'?'SUPERVISED FINE-TUNING':'GROUP-RELATIVE RL'}</span><strong className="accuracy">{m?(m.accuracy*100).toFixed(1)+'%':'—'}</strong><div className="meter"><i style={{width:`${(m?.accuracy||0)*100}%`}}/></div><p>{m?`${m.correct} / ${m.n} held-out routing tasks`:'Training results pending'}</p><dl><dt>Mean routing latency</dt><dd>{m?m.mean_latency_seconds.toFixed(2)+'s':'—'}</dd><dt>Estimated $ / correct</dt><dd>{m?'$'+m.estimated_dollars_per_correct.toFixed(6):'—'}</dd></dl></article>})}</div><div className="methodology"><h2>What these numbers mean</h2><p>All {evaluation.split_sizes?Object.values(evaluation.split_sizes).reduce((a,b)=>a+b,0):0} questions are synthetic. Train, validation and test use different question-template families. Candidate selection happens on validation; the held-out test is run afterward. Small samples and synthetic coverage limit what these scores establish.</p><p>Cost estimates use $0.04 per CPU-hour multiplied by observed routing time, divided by correct answers. They exclude hosting idle time, video indexing, storage and networking. They are not an invoice or a full product cost.</p><p>GRPO samples a group of six actions from a constrained five-plan policy, uses exact routing rewards and a small tool-cost penalty, and applies clipped updates with an SFT reference. It does not train visual embeddings or a free-form reasoning model.</p><a href="https://github.com/Eunini/scenetrace/tree/main/artifacts" target="_blank" rel="noreferrer">Read the full evaluation and training record <ArrowUpRight size={15}/></a></div></>}</section>}
        {tab==='about'&&<section className="about"><div className="eyebrow">A WORKSPACE WITH BOUNDARIES</div><h1>Search the footage.<br/><span>Verify the finding.</span></h1><p className="lede">SceneTrace helps you find candidate moments quickly. It keeps the source close so you can decide whether the result is useful.</p><div className="about-grid"><article><FilmStrip size={26}/><h2>What gets indexed</h2><p>Up to 32 evenly spaced frames per video. A frozen CLIP image/text encoder ranks these frames against your search. Events between samples can be missed; there is no audio transcription or exhaustive video analysis.</p></article><article><ShieldCheck size={26}/><h2>Your visitor workspace</h2><p>Uploads are processed on the server, isolated by a signed, HTTP-only visitor cookie. They expire after 24 hours and may disappear sooner on a host restart. No sign-up, advertising tracker, or automatic training on your uploads.</p></article><article><Scissors size={26}/><h2>You approve the export</h2><p>A request prepares a time range. Rendering starts only after you approve it. Export up to 30 seconds as an MP4; download before leaving. Delete individual videos or clear the entire workspace.</p></article><article><ChartBar size={26}/><h2>A measured agent</h2><p>An open-weight SmolLM2 router chooses among five tool plans. Requests have a four-call budget, typed inputs, bounded context and recoverable failures. The Evaluation page shows the actual training results and their limitations.</p></article></div><div className="methodology"><h2>Before you upload</h2><p>Use MP4, MOV or WebM, up to 50 MB and three minutes. Keep a copy of your original. This public CPU service has a shared queue and can take a few minutes on first use. It is an early release; use non-sensitive footage you have permission to process.</p><p>Object detection covers the model’s COCO categories. Foreground masks use GrabCut initialized from a box. Neither is a verified inventory of the scene.</p><h2>Open-film sample</h2><p>Big Buck Bunny (2008). (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org. Excerpted, resized and re-encoded under <a href="https://creativecommons.org/licenses/by/3.0/" target="_blank" rel="noreferrer">CC BY 3.0</a>. <a href="https://peach.blender.org/about/" target="_blank" rel="noreferrer">Original attribution and license</a>. No endorsement is implied.</p></div></section>}
        <footer><span><FilmStrip size={15}/> SceneTrace <i/> Built by Inioluwa Atanda</span><span>Evidence before answers.<a href="https://github.com/Eunini/scenetrace" target="_blank" rel="noreferrer">GitHub <ArrowUpRight size={12}/></a></span></footer>
      </main>
    </div>
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
