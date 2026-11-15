const $=id=>document.getElementById(id);
let ws,context,stream,capture,source,active=false,run=0,scheduled=0,playing=new Set(),finishTimer;

const replyLines=new Map();
function status(text){$('status').textContent=text;}
function error(text){$('error').textContent=text;$('error').hidden=!text;}
function interrupt(){clearTimeout(finishTimer);for(const node of playing){try{node.stop();}catch{}}playing.clear();scheduled=context?.currentTime||0;}
async function stop(){
  const stopped=++run;active=false;document.body.classList.remove('active');$('start').textContent='Start conversation';$('start').disabled=true;
  const socket=ws;ws=null;if(socket){socket.onclose=null;socket.close();}
  stream?.getTracks().forEach(t=>t.stop());stream=null;
  capture?.disconnect();source?.disconnect();capture=null;source=null;
  interrupt();const old=context;context=null;
  if(old)await old.close();
  if(stopped===run){$('start').disabled=false;status('Ready');}
}
async function fail(text){await stop();error(text);}
function line(role,text,turn,exact=false){
  if(!text)return;
  const log=$('conversation');
  const follow=log.scrollHeight-log.scrollTop-log.clientHeight<48;
  if(turn&&replyLines.has(turn)){
    replyLines.get(turn).textContent += (exact ? "" : " ") + text;
  }else{
    $('empty')?.remove();
    const row=document.createElement('article');row.className='line '+(role==='user'?'user':'assistant');
    const name=document.createElement('div');name.className='role';name.textContent=role==='user'?'You':'Vexo';
    const body=document.createElement('p');body.textContent=text;
    row.append(name,body);log.append(row);if(turn)replyLines.set(turn,body);
  }
  if(follow)log.scrollTop=log.scrollHeight;
}

function play(data){
  if(!context||!active)return;const samples=new Int16Array(data);const buffer=context.createBuffer(1,samples.length,24000);
  const channel=buffer.getChannelData(0);for(let i=0;i<samples.length;i++)channel[i]=samples[i]/32768;
  const node=context.createBufferSource();node.buffer=buffer;node.connect(context.destination);playing.add(node);node.onended=()=>playing.delete(node);
  scheduled=Math.max(context.currentTime+.03,scheduled);node.start(scheduled);scheduled+=buffer.duration;
}
async function start(){
  error('');$('start').disabled=true;status('Connecting');const current=++run;
  try{
    context=new AudioContext({sampleRate:24000});await context.resume();
    if(context.sampleRate!==24000)throw new Error('This browser cannot capture at 24 kHz. Use Chrome.');
    {
    const incoming=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:false,autoGainControl:false},video:false});
    if(current!==run){incoming.getTracks().forEach(t=>t.stop());return;}stream=incoming;
    for(const track of stream.getTracks())track.onended=()=>{if(current===run)fail('Microphone disconnected. Please reconnect.');};
    await context.audioWorklet.addModule('/static/capture.js');
    if(current!==run)return;
    source=context.createMediaStreamSource(stream);capture=new AudioWorkletNode(context,'capture');
    }
    ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/conversation`);ws.binaryType='arraybuffer';
    const socket=ws;
    ws.onopen=()=>{if(current===run)socket.send(JSON.stringify({prediction:$('prediction').checked,pause_ms:300}));};
    ws.onmessage=async event=>{
      if(current!==run)return;
      if(event.data instanceof ArrayBuffer){play(event.data);return;}
      const message=JSON.parse(event.data);
      switch(message.type){
        case 'ready':
          active=true;document.body.classList.add('active');$('start').textContent='End conversation';$('start').disabled=false;status('Listening');
          const send=pcm=>{if(current===run&&socket.readyState===WebSocket.OPEN){if(socket.bufferedAmount>24000*2)void fail('Audio connection fell behind. Please reconnect.');else socket.send(pcm);}};
          capture.port.onmessage=e=>send(e.data);source.connect(capture);capture.connect(context.destination);
          break;
        case 'transcript':line(message.role,message.text,message.turn_id,message.append_exact);break;
        case 'analysis_error':$('score').textContent='—';$('level').textContent='Analysis unavailable';break;
        case 'prediction':
          $('score').textContent=message.score.toFixed(1);$('level').textContent=message.level;$('level').className=message.level;
          const outside=message.score<0||message.score>10;$('rangeWarning').hidden=!outside;$('scale').hidden=outside;break;
        case 'event':break;
        case 'status':status(message.text);break;
        case 'interrupt':interrupt();status('Listening');break;
        case 'audio_done':clearTimeout(finishTimer);finishTimer=setTimeout(()=>{if(current===run)status('Listening');},Math.max(0,(scheduled-context.currentTime)*1000));break;
        case 'error':fail(message.text);break;
      }
    };
    ws.onerror=()=>{if(current===run)fail('Could not connect to the demo server.');};
    ws.onclose=()=>{if(current===run)fail('Connection closed. Start a new conversation.');};
  }catch(e){if(current===run)await fail(e.name==='NotAllowedError'?'Allow microphone access to start the conversation.':e.message);}
}
$('start').onclick=()=>active?stop():start();
$('prediction').onchange=async()=>{
  const restart=active;await stop();$('score').textContent='—';$('level').textContent='';$('rangeWarning').hidden=true;$('scale').hidden=false;
  $('scoreCard').hidden=!$('prediction').checked;$('modeDescription').textContent=$('prediction').checked?'Lamita · ResNet + reference':'Normal conversation';
  replyLines.clear();$('conversation').replaceChildren();const p=document.createElement('p');p.id='empty';p.className='empty';p.textContent='Speak or play your recording into the microphone.';$('conversation').append(p);
  if(restart)await start();
};
window.addEventListener('pagehide',()=>{stream?.getTracks().forEach(t=>t.stop());ws?.close();});



fetch('/health').then(r=>r.json()).then(h=>{if(!h.pain_model_ready){$('prediction').disabled=true;$('modeDescription').textContent='Connect a pain model to enable prediction';}}).catch(()=>error('Start the voice server to connect.'));
