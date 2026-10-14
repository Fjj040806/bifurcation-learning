/* Functional DOM-mock test, not a browser visual test. */
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('assert');
const root=path.resolve(__dirname,'..'),fixture=JSON.parse(fs.readFileSync(path.join(__dirname,'demo_expected.json'),'utf8'));
const html=fs.readFileSync(path.join(root,'demo/index.html'),'utf8'),inline=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements={};
for(const id of ['truth','learned','system','parameter','pause','reset','paramName','paramValue','trueCritical','learnedCritical','clock']){
 elements[id]={id,clientWidth:550,clientHeight:420,width:550,height:420,listeners:{},textContent:'',value:'.1',
   addEventListener(name,handler){this.listeners[name]=handler;},
   getContext(){return {fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){}};}};
}
const sandbox={document:{getElementById:id=>elements[id]},window:{addEventListener(){}},devicePixelRatio:1,requestAnimationFrame(){},console};
vm.createContext(sandbox);
vm.runInContext(inline,sandbox);
for(const f of fixture){
 const result=vm.runInContext('system='+JSON.stringify(f.system)+';parameter='+f.parameter+';field('+JSON.stringify(f.state)+',true)',sandbox);
 for(let i=0;i<2;i++)assert(Math.abs(result[i]-f.expected[i])<1e-10,'JS/Python model mismatch');
}
elements.system.listeners.change({target:{value:'fhn'}});
assert.strictEqual(elements.paramName.textContent,'I');assert.strictEqual(elements.trueCritical.textContent,'0.3313');
elements.parameter.listeners.input({target:{value:'.4'}});
assert.strictEqual(elements.paramValue.textContent,'0.400');
elements.pause.onclick();assert.strictEqual(elements.pause.textContent,'Resume');
elements.reset.onclick();assert.strictEqual(vm.runInContext('time',sandbox),0);
console.log('Demo functional checks passed: model outputs, controls and restart. Browser visual QA not performed.');
