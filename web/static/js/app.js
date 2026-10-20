'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const money = n => new Intl.NumberFormat('en-IN', {style:'currency',currency:'INR',maximumFractionDigits:0}).format(n);
const state = {hubs:[], custom:null, data:null, selected:'all', busy:false, key:sessionStorage.getItem('parcel-key') || '', fleet:{num_bikes:2,num_petrol_trucks:1,num_diesel_trucks:8}};
let map, routesLayer, nodesLayer;
function notice(message, success=false) { $('notice').hidden=!message; $('notice').textContent=message; $('notice').className=success?'success':''; }
async function api(path, body) {
  const controller = new AbortController(); const timeout=setTimeout(()=>controller.abort(),45000);
  try {
    const response=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json','X-API-Key':state.key},body:body?JSON.stringify(body):undefined,signal:controller.signal});
    const data=await response.json();
    if(!response.ok) throw new Error(typeof data.detail==='string'?data.detail:`Request failed (${response.status})`);
    return data;
  } catch(error) { if(error.name==='AbortError') throw new Error('The server took too long. Check your connection and try again.'); throw error; }
  finally {clearTimeout(timeout);}
}
function tab(name) {
  document.querySelectorAll('.tab-panel').forEach(el=>el.hidden=el.id!==name);
  document.querySelectorAll('[data-tab]').forEach(el=>el.classList.toggle('active',el.dataset.tab===name));
  if(name==='overview') setTimeout(()=>map?.invalidateSize(),20);
  if(name==='history') loadHistory().catch(e=>notice(e.message));
}
async function initMap() {
  if(!window.L) { $('map').textContent='Map could not load. Route manifests remain available.'; return; }
  map=L.map('map',{scrollWheelZoom:false}).setView([11.0168,76.98],12);
  let config={tile_url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png',attribution:'&copy; OpenStreetMap contributors'};
  try { config=await api('/api/map-config'); } catch { /* Default interactive pilot tiles. */ }
  const tiles=L.tileLayer(config.tile_url,{attribution:config.attribution,maxZoom:19}).addTo(map);
  tiles.on('tileerror',()=>{$('map-source').textContent='Map tiles unavailable · route manifests remain available';});
  routesLayer=L.layerGroup().addTo(map); nodesLayer=L.layerGroup().addTo(map); drawMap();
}
function drawMap() {
  if(!map || !routesLayer || !nodesLayer) return;
  routesLayer.clearLayers();nodesLayer.clearLayers();
  const hubs=state.data?.hubs || state.hubs;
  const selected=state.data?.vehicle_manifests.find(v=>String(v.vehicle_id)===state.selected);
  const allowed=selected?new Set(selected.route_stops):null;
  const bounds=[];
  for(const h of hubs) {
    if(allowed && !allowed.has(h.id)) continue;
    const ll=[h.lat,h.lon];bounds.push(ll);
    L.marker(ll,{icon:L.divIcon({className:`parcel-marker ${h.id===0?'depot-marker':''}`,html:h.id===0?'p.':esc(h.id),iconSize:[23,23]})}).addTo(nodesLayer).bindPopup(esc(h.name));
  }
  for(const p of state.data?.polylines || []) if(state.selected==='all' || String(p.route_id)===state.selected) L.polyline(p.polyline,{color:p.color,weight:3,opacity:.8,dashArray:'5,5'}).addTo(routesLayer);
  if(bounds.length) map.fitBounds(bounds,{padding:[30,30],maxZoom:14});
}
function renderOrders() {
  const q=$('search').value.toLowerCase();
  const orders=state.hubs.filter(h=>h.id!==0);
  $('order-count').textContent=orders.length;
  $('dataset-label').textContent=`${state.custom?.name || 'Coimbatore sample'} · ${orders.length} orders · ${orders.reduce((a,h)=>a+h.demand,0)} kg`;
  $('order-rows').innerHTML=orders.filter(h=>[h.name,h.address,h.customer_name,h.product].join(' ').toLowerCase().includes(q)).map(h=>`<tr><td>#${esc(h.id)}</td><td>${esc(h.customer_name || h.name)}<small>${esc(h.address || h.name)}</small></td><td>${esc(h.product || 'Parcel')}<small>${esc(h.size)}</small></td><td>${h.demand} kg</td></tr>`).join('') || '<tr><td colspan="4">No matching orders.</td></tr>';
  capacityNote();
}
function capacityNote() {
  const capacity=state.fleet.num_bikes*15+state.fleet.num_petrol_trucks*60+state.fleet.num_diesel_trucks*80;
  const demand=state.hubs.reduce((s,h)=>s+h.demand,0);
  $('capacity-note').textContent=`${capacity} kg available · ${demand} kg of parcels${demand>capacity?' · Add vehicles to cover the full load.':''}`;
}
function renderFleet() {
  $('fleet-controls').innerHTML=[['num_bikes','City bikes','15 kg · compact deliveries'],['num_petrol_trucks','Petrol vans','60 kg · everyday parcels'],['num_diesel_trucks','Diesel vans','80 kg · the heavier things']].map(([key,name,sub])=>`<div class="fleet-row"><div><h3>${name}</h3><p>${sub}</p></div><div class="stepper"><button type="button" data-step="${key}" data-delta="-1" aria-label="Remove ${name}">−</button><output>${state.fleet[key]}</output><button type="button" data-step="${key}" data-delta="1" aria-label="Add ${name}">+</button></div></div>`).join('');
  capacityNote();
}
function selectVehicle(id) {state.selected=String(id);$('route-filter').value=state.selected;renderDispatch();drawMap();}
function renderDispatch() {
  const d=state.data;if(!d)return;
  const s=d.fleet_summary;
  $('stats').innerHTML=[['Orders assigned',`${s.assigned_orders}<i>/ ${d.num_customers}</i>`,`${s.total_vehicles_used} vehicles in your plan`],['Planned distance',`${s.total_distance_km}<i>km</i>`,'Estimated across all routes'],['Estimated fuel cost',money(s.total_cost_inr),`${s.total_fuel_liters} litres · planning rates`],['Delivered',`${s.delivered_orders}<i>/ ${s.assigned_orders}</i>`,`${s.failed_orders} failed · ${s.unassigned_orders} unassigned`]].map(([label,value,note])=>`<div class="stat"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('');
  $('route-filter').innerHTML='<option value="all">All vehicles</option>'+d.vehicle_manifests.map(v=>`<option value="${v.vehicle_id}">${esc(v.vehicle_name)}</option>`).join('');$('route-filter').value=state.selected;
  $('unassigned').innerHTML=d.unassigned_orders.length?`<div class="warning"><b>${d.unassigned_orders.length} orders need a vehicle.</b> Add capacity and create a new plan.<details><summary>View parcels</summary>${d.unassigned_orders.map(o=>`<p>#${o.customer_id} · ${esc(o.name)} · ${o.demand_kg} kg — ${esc(o.reason)}</p>`).join('')}</details></div>`:'';
  $('routes').innerHTML=d.vehicle_manifests.map(v=>{const done=v.schedule.filter(s=>s.status==='delivered').length;return `<article tabindex="0" role="button" aria-label="View ${esc(v.vehicle_name)}" data-vehicle="${v.vehicle_id}" class="card route-card ${state.selected===String(v.vehicle_id)?'selected':''}"><div class="route-top"><span class="vehicle-icon">${v.vehicle_type==='bike'?'⌁':'▱'}</span><span class="tag">${v.pending_ids.length?'READY TO GO':'ROUTE REVIEWED'}</span></div><h3>${esc(v.vehicle_name)} <span style="color:${v.color}">↗</span></h3><p>${v.num_stops} stops · ${v.total_payload_kg} / ${v.capacity_kg} kg capacity</p><div class="route-metrics"><span>${v.distance_km.toFixed(1)}<small>kilometres</small></span><span>${money(v.fuel_cost_inr)}<small>estimated fuel</small></span><span>${v.fuel_liters.toFixed(2)} L<small>consumption</small></span></div><div class="progress-track"><i style="width:${done/v.num_stops*100}%"></i></div><p>${done} delivered · ${v.pending_ids.length} stops to go</p></article>`;}).join('') || '<div class="empty-state">No parcels fit the selected fleet. Add a larger vehicle to continue.</div>';
  const v=d.vehicle_manifests.find(v=>String(v.vehicle_id)===state.selected) || d.vehicle_manifests.find(v=>v.pending_ids.length);
  const stop=v?.schedule.find(s=>s.customer_id===v.pending_ids[0]);
  $('itinerary').innerHTML=v?`<div class="section-heading"><div><span class="eyebrow">EVERY DOORSTEP</span><h2>${esc(v.vehicle_name)} · itinerary</h2></div></div><div class="card table-wrap"><table><thead><tr><th>Stop</th><th>Recipient & parcel</th><th>Estimated arrival</th><th>Delivery</th></tr></thead><tbody>${[...v.schedule.filter(s=>!v.pending_ids.includes(s.customer_id)),...v.pending_ids.map(id=>v.schedule.find(s=>s.customer_id===id))].map(s=>`<tr><td>#${s.customer_id}</td><td>${esc(s.customer_name || s.name)}<small>${esc(s.product)} · ${s.demand_kg} kg · ${esc(s.address || s.name)}</small></td><td>${Math.floor((480+s.arrival_time_min)/60)%24}:${String(Math.floor(s.arrival_time_min)%60).padStart(2,'0')}<small>${s.late_minutes>0?`${s.late_minutes} min after window`:'Within window'}</small></td><td>${esc(s.status)}${s.updated_at?`<small>${esc(new Date(s.updated_at).toLocaleTimeString())}</small>`:''}</td></tr>`).join('')}</tbody></table></div>`:'';
  $('next-card').innerHTML=stop?`<span class="eyebrow">NEXT DOORSTEP · ${esc(v.vehicle_name)}</span><span class="tag" style="margin-top:20px">STOP ${stop.stop_number} · ${stop.status==='retry'?'RETRY':'PENDING'}</span><h2 class="stop-title">${esc(stop.customer_name || stop.name)}</h2><p>${esc(stop.address || stop.name)}<br>${esc(stop.product)} · ${stop.demand_kg} kg</p><div class="next-meta"><span>Est. ${Math.floor((480+stop.arrival_time_min)/60)%24}:${String(Math.round(stop.arrival_time_min)%60).padStart(2,'0')}</span><span>Onboard ${v.remaining_payload_kg} kg</span></div><div class="stop-links"><a href="https://www.google.com/maps/dir/?api=1&destination=${stop.lat},${stop.lon}" target="_blank" rel="noopener">Open directions ↗</a>${stop.phone?`<a href="tel:${encodeURIComponent(stop.phone)}">Call customer</a>`:''}</div><div class="action-stack"><button class="button primary full" data-event="delivered" data-vid="${v.vehicle_id}" data-cid="${stop.customer_id}" ${state.busy?'disabled':''}>Mark delivered ✓</button>${stop.attempts===0?`<button class="text-button" data-event="retry" data-vid="${v.vehicle_id}" data-cid="${stop.customer_id}" ${state.busy?'disabled':''}>Unavailable? Retry at the end ↗</button>`:''}<button class="text-button" data-event="failed" data-vid="${v.vehicle_id}" data-cid="${stop.customer_id}" ${state.busy?'disabled':''}>Unable to deliver · return parcel</button></div><p class="fine-print">Confirm only after reaching the customer. Failed parcels stay onboard.</p>`:`<span class="eyebrow">ALL STOPS REVIEWED</span><div class="empty-illustration">✓</div><h2>A good day,<br>well delivered.</h2><p>${s.failed_orders?'Return failed parcels to the depot.':'All assigned deliveries are complete.'}${s.unassigned_orders?' Unassigned orders still need planning.':''}</p>`;
  $('comparison').innerHTML=`<div class="results-table"><span class="eyebrow">PLANNING NOTES</span><p class="fine-print">${esc(d.distance_source)}. ${esc(d.fuel_source)}. Computed in ${(d.solve_time_sec*1000).toFixed(0)} ms. ${esc(d.warning || '')}</p>${Object.keys(d.results).length>1?`<div class="card table-wrap"><table><thead><tr><th>Engine</th><th>Distance</th><th>Unassigned</th><th>Time</th></tr></thead><tbody>${Object.entries(d.results).map(([k,r])=>`<tr><td>${esc(k)}</td><td>${r.error?esc(r.error):`${r.distance_km} km`}</td><td>${r.unassigned??'—'}</td><td>${r.solve_time_sec!=null?(r.solve_time_sec*1000).toFixed(0)+' ms':'—'}</td></tr>`).join('')}</tbody></table></div>`:''}</div>`;
}
async function refresh() {if(!state.data)return;state.data=await api(`/api/dispatches/${state.data.dispatch_id}`);renderDispatch();drawMap();}
async function submitPlan(event) {
  event?.preventDefault();if(state.busy)return;
  state.busy=true;$('plan-button').disabled=true;$('plan-top').disabled=true;$('plan-button').textContent='Finding a thoughtful route…';notice('');
  try {
    const form=new FormData($('planning-form'));
    state.data=await api('/api/solve',{dataset_type:state.custom?'custom':'coimbatore',custom_data:state.custom,num_customers:50,algorithm:form.get('algorithm'),fleet_config:{...state.fleet,petrol_price_inr:Number(form.get('petrol_price_inr')),diesel_price_inr:Number(form.get('diesel_price_inr'))},weight_dist:Number($('weight_dist').value),weight_fuel:Number($('weight_fuel').value),weight_tw:Number($('weight_tw').value)});
    localStorage.setItem('parcel-dispatch',state.data.dispatch_id);state.selected='all';renderDispatch();tab('overview');drawMap();notice('Your dispatch is saved. Choose a vehicle to start delivering.',true);
  } catch(e){notice(e.message);} finally{state.busy=false;$('plan-button').disabled=false;$('plan-top').disabled=false;$('plan-button').innerHTML='Create dispatch plan <span>↗</span>';}
}
async function deliveryEvent(button) {
  if(state.busy)return;state.busy=true;renderDispatch();
  const event={event_id:crypto.randomUUID(),expected_version:state.data.version,vehicle_id:Number(button.dataset.vid),customer_id:Number(button.dataset.cid),action:button.dataset.event};
  try{state.data=await api(`/api/dispatches/${state.data.dispatch_id}/events`,event);notice(event.action==='retry'?'Remaining stops reordered from this location; parcel retained for retry.':'Delivery progress saved.',true);}
  catch(e){notice(e.message);try{await refresh();}catch{/* Error already visible. */}}
  finally{state.busy=false;renderDispatch();drawMap();}
}
async function loadSample(){const d=await api('/api/sample-50-orders');state.hubs=d.orders;state.custom=null;renderOrders();drawMap();}
async function loadHistory(){const list=await api('/api/dispatches');$('history-list').innerHTML=list.map(d=>`<button class="card route-card" data-dispatch="${d.dispatch_id}" style="text-align:left;color:inherit"><span class="eyebrow">${esc(new Date(d.created_at).toLocaleString())}</span><h2>${esc(d.dataset_name)}</h2><p>${d.fleet_summary.delivered_orders} delivered · ${d.fleet_summary.total_distance_km} km planned</p><span class="text-button">Continue this dispatch ↗</span></button>`).join('')||'<div class="empty-state">Your first dispatch is waiting to happen.</div>';}
async function connect(){try{await api('/api/health');await loadSample();$('connection').textContent='Your workspace is connected';$('connection-dot').style.background='#7b9c7e';const id=localStorage.getItem('parcel-dispatch');if(id){try{state.data=await api(`/api/dispatches/${id}`);renderDispatch();drawMap();}catch(e){notice(e.message);}}}catch(e){$('connection').textContent='Connection needs attention';$('connection-dot').style.background='#c18a6b';notice(e.message);}}
document.addEventListener('DOMContentLoaded',()=>{
  $('today').textContent=new Date().toLocaleDateString('en-IN',{weekday:'long',day:'numeric',month:'long'});
  $('weights').innerHTML=[['weight_dist','A shorter journey',.5],['weight_fuel','A lighter footprint',.3],['weight_tw','Closer to the window',.2]].map(([id,label,val])=>`<div class="weight-control"><label for="${id}">${label}<output id="${id}-out">${Math.round(val*100)}%</output></label><input id="${id}" type="range" min="0" max="1" step=".05" value="${val}"></div>`).join('');
  renderFleet();initMap();connect();
  document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>tab(b.dataset.tab));document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>tab(b.dataset.go));
  $('planning-form').onsubmit=submitPlan;$('plan-top').onclick=()=>tab('fleet');$('search').oninput=renderOrders;
  $('fleet-controls').onclick=e=>{const b=e.target.closest('[data-step]');if(b){state.fleet[b.dataset.step]=Math.max(0,Math.min(30,state.fleet[b.dataset.step]+Number(b.dataset.delta)));renderFleet();}};
  $('weights').oninput=e=>{if(e.target.type==='range')$(e.target.id+'-out').textContent=Math.round(Number(e.target.value)*100)+'%';};
  $('route-filter').onchange=e=>selectVehicle(e.target.value);
  $('routes').onclick=e=>{const c=e.target.closest('[data-vehicle]');if(c)selectVehicle(c.dataset.vehicle);};
  $('routes').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.target.click();}};
  $('next-card').onclick=e=>{const b=e.target.closest('[data-event]');if(b)deliveryEvent(b);};
  $('refresh').onclick=()=>refresh().catch(e=>notice(e.message));$('load-sample').onclick=()=>loadSample().catch(e=>notice(e.message));
  $('csv-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>500000)throw new Error('CSV exceeds 500 KB');state.custom=await api('/api/import-csv',{content:await file.text(),name:file.name.replace(/\.csv$/i,'')});state.hubs=state.custom.hubs;renderOrders();notice('Orders imported. Review your fleet before planning.',true);}catch(err){notice(err.message);}e.target.value='';};
  $('download-sample').onclick=async()=>{try{const r=await fetch('/api/sample.csv',{headers:{'X-API-Key':state.key}});if(!r.ok)throw new Error('Could not download CSV');const url=URL.createObjectURL(await r.blob());const a=document.createElement('a');a.href=url;a.download='coimbatore-orders.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){notice(e.message);}};
  $('settings-open').onclick=()=>{$('access-key').value=state.key;$('settings-dialog').showModal();};$('settings-close').onclick=()=>$('settings-dialog').close();
  $('settings-form').onsubmit=e=>{e.preventDefault();state.key=$('access-key').value.trim();sessionStorage.setItem('parcel-key',state.key);$('settings-dialog').close();connect();};
  $('history-list').onclick=async e=>{const b=e.target.closest('[data-dispatch]');if(!b)return;try{state.data=await api(`/api/dispatches/${b.dataset.dispatch}`);localStorage.setItem('parcel-dispatch',state.data.dispatch_id);state.selected='all';renderDispatch();tab('overview');drawMap();}catch(err){notice(err.message);}};
});
