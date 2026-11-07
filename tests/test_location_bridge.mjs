// Pure runtime tests: never request real device location or operate a browser.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source = readFileSync(new URL('../forecast_exe/location_bridge.js', import.meta.url), 'utf8')
  .replace('export default function', 'function render')+'\nrender;';
const city = {name:'London, England, United Kingdom',latitude:51.50853,longitude:-0.12574,timezone:'Europe/London'};
function mount({saved=null, denied=false, insecure=false, blocked=false, write=null}={}) {
  const nodes = Object.fromEntries(['[data-use]','[data-forget]','[data-remember]','[data-status]']
    .map(k=>[k,{disabled:false,checked:true,textContent:''}]));
  const values = new Map(saved ? [['forecast.exe.preferred-city.v1',JSON.stringify({city:saved})]] : []);
  const events = [];
  let calls = 0;
  const context = vm.createContext({
    window:{isSecureContext:!insecure}, crypto:{randomUUID:()=>String(events.length+1)},
    localStorage:{getItem:k=>{if(blocked)throw Error('blocked');return values.get(k)||null;},
      setItem:(k,v)=>{if(blocked)throw Error('blocked');values.set(k,v);},removeItem:k=>values.delete(k)},
    navigator:{geolocation:{getCurrentPosition:(ok,fail,options)=>{
      calls++; assert.equal(options.enableHighAccuracy,false); assert.equal(options.timeout,12000);
      denied ? fail({code:1}) : ok({coords:{latitude:40.712834,longitude:-74.006078}});
    }}}
  });
  const cleanup = vm.runInContext(source,context)({parentElement:{querySelector:k=>nodes[k]},
    data:{write,remember:true},setTriggerValue:(key,value)=>{assert.equal(key,'event');events.push(value);}});
  return {nodes,events,values,cleanup,get calls(){return calls;}};
}
const first = mount();
assert.equal(first.calls,0,'Page load must never request geolocation');
first.nodes['[data-use]'].onclick();
assert.equal(first.calls,1);
assert.equal(first.events.at(-1).latitude,40.71);
assert.equal(first.events.at(-1).longitude,-74.01);
assert.equal(first.events.at(-1).kind,'detected');
assert.equal(first.values.size,0,'Raw device coordinates must never enter persistent storage');
const denied = mount({denied:true});
denied.nodes['[data-use]'].onclick();
assert.match(denied.events.at(-1).message,/declined/);
assert.equal(denied.nodes['[data-use]'].disabled,false);
const restored = mount({saved:city});
assert.equal(restored.calls,0);
assert.equal(restored.events[0].kind,'restore');
const persisted = mount({write:{city,remember:true,revision:'one'}});
assert.equal(JSON.parse(persisted.values.get('forecast.exe.preferred-city.v1')).city.name,city.name);
persisted.nodes['[data-forget]'].onclick();
assert.equal(persisted.values.size,0);
assert.equal(persisted.events.at(-1).kind,'forget');
const insecure = mount({insecure:true});
insecure.nodes['[data-use]'].onclick();
assert.equal(insecure.calls,0);
assert.match(insecure.events.at(-1).message,/unavailable/);
const blocked = mount({blocked:true});
assert.match(blocked.nodes['[data-status]'].textContent,/storage is unavailable/);
console.log('Location bridge: no automatic permission request, rounded payload, denial fallback, restoration, persistence, forget, and blocked storage verified.');
