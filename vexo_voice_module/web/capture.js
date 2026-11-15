class Capture extends AudioWorkletProcessor {
  constructor(){super();this.samples=new Int16Array(480);this.index=0;}
  process(inputs,outputs){
    const input=inputs[0]?.[0];
    if(input)for(const value of input){
      this.samples[this.index++]=Math.max(-32768,Math.min(32767,Math.round(value*32768)));
      if(this.index===480){this.port.postMessage(this.samples.buffer,[this.samples.buffer]);this.samples=new Int16Array(480);this.index=0;}
    }
    return true;
  }
}
registerProcessor('capture',Capture);
