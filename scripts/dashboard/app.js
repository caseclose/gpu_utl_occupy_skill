"use strict";
const $ = id => document.getElementById(id);
const escape = text => String(text ?? "—").replace(/[&<>"']/g, c => ({
  "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"
})[c]);
const num = (n, digits=0) => Number.isFinite(n) ? n.toFixed(digits) : "—";
const percent = n => Number.isFinite(n) ? num(n) + "%" : "—";
const memory = n => n == null ? "—" : n >= 1024 ? num(n/1024, 1) + " GiB" : n + " MiB";
const clock = new Intl.DateTimeFormat("zh-CN", {timeZone:"Asia/Shanghai", hour:"2-digit", minute:"2-digit", second:"2-digit", hourCycle:"h23"});
const dated = new Intl.DateTimeFormat("zh-CN", {timeZone:"Asia/Shanghai", month:"2-digit", day:"2-digit", hour:"2-digit", minute:"2-digit", second:"2-digit", hourCycle:"h23"});
const when = (t, format=clock) => t ? format.format(new Date(t * 1000)) : "—";
const avg = values => {
  const known = values.filter(Number.isFinite);
  return known.length ? known.reduce((a,b)=>a+b,0)/known.length : null;
};
const sum = values => values.filter(Number.isFinite).reduce((a,b)=>a+b,0);
const scope = {snapshot:{nodes:[]}, node:0, card:0, history:false, posting:false, reading:false, pending:null, disconnected:false};
const selections = new Map();
const stale = node => !node?.latest || node.connection_error || Date.now()/1000 - node.latest.time > 30;
const current = () => scope.snapshot.nodes[scope.node];
function chosen() {
  const name=current()?.name;
  if (!selections.has(name)) selections.set(name,new Set());
  return selections.get(name);
}
const count = (el, value) => { el.textContent = value; };
const deviceLabel = card => card.label ?? "GPU " + card.index;
const ratio = (used, total) => used == null || !total ? null : 100*used/total;
const meter = (value,label) => `<div class="reading"><strong>${escape(label)}</strong><span class="meter-track"><i style="width:${Math.min(100,Math.max(0,value||0))}%"></i></span></div>`;
const classFor = category => ({占卡:"keepalive",训练:"training",推理:"inference"})[category] || "unknown";


const nodeKey=node=>node.name;
const nodeNotes=new Map();
const nodeNotesKey="gpu-dashboard-node-notes";
try {
  const saved=JSON.parse(localStorage.getItem(nodeNotesKey) || "{}");
  for (const [id,note] of Object.entries(saved)) {
    if (typeof note==="string" && note.length<=40 && note.trim()) nodeNotes.set(id,note.trim());
  }
} catch {}

function nodeTitle(node) {
  return nodeNotes.get(nodeKey(node)) ?? node.name;
}

function renderOverview(nodes) {
  $("overview-rows").innerHTML=nodes.length ? nodes.map(node=>{
    const gpus=node.latest?.gpus ?? [];
    const fresh=!stale(node);
    const state=fresh?"在线":node.latest?"数据过期":"等待采集";
    const installed=gpus.length;
    const used=gpus.map(g=>g.used).filter(Number.isFinite);
    const workerCount=state=>gpus.filter(g=>g.worker_state===state).length;
    return `<tr><td><strong>${escape(nodeTitle(node))}</strong></td>
      <td><span class="state ${fresh?"":"late"}">${state}</span></td>
      <td>${fresh?gpus.length:0} / ${installed||"—"}</td>
      <td>${fresh?percent(avg(gpus.map(g=>g.util))):"—"}</td>
      <td>${fresh&&used.length?memory(sum(used)):"—"}</td>
      <td>${fresh?workerCount("running")+" / "+workerCount("yielding"):"—"}</td>
      <td class="mono">${when(node.latest?.time)}</td></tr>`;
  }).join("") : '<tr><td colspan="7" class="empty-cell">正在等待节点数据</td></tr>';
}

const noteDialog=$("node-note-dialog");
const noteInput=$("node-note-input");
let editedNode=null;
function editNote(node) {
  editedNode=nodeKey(node);
  count($("node-note-id"),editedNode);
  noteInput.value=nodeNotes.get(editedNode) ?? "";
  noteDialog.showModal();
  noteInput.select();
}
noteDialog.querySelector("form").addEventListener("submit",event=>{
  if (event.submitter?.value!=="save" || editedNode===null) return;
  const note=noteInput.value.trim();
  if (note) nodeNotes.set(editedNode,note); else nodeNotes.delete(editedNode);
  try { localStorage.setItem(nodeNotesKey,JSON.stringify(Object.fromEntries(nodeNotes))); } catch {}
  renderNodes();
});

function renderNodes() {
  const nodes = scope.snapshot.nodes;
  const available = nodes.filter(n=>!stale(n));
  const all = available.flatMap(n=>n.latest.gpus);
  const totals = [all.length + " / " + sum(nodes.map(n=>n.latest?.gpus.length)), percent(avg(all.map(g=>g.util))),
    all.length ? memory(sum(all.map(g=>g.used))) : "—",
    available.length ? sum(available.map(n=>n.latest.processes.length)) : "—"];
  $("summary").querySelectorAll(".stat-v").forEach((e,i)=>{e.textContent=totals[i];});
  $("nodes").innerHTML = nodes.map((n,i)=>{
    const gpus=n.latest?.gpus || [];
    const state=stale(n)?"数据过期":"在线";
    return `<div class="node-tile"><button class="node" type="button" data-node="${i}" aria-pressed="${i===scope.node}">
      <span class="node-top"><strong title="${escape(nodeTitle(n))}">${escape(nodeTitle(n))}</strong><span class="state ${stale(n)?"late":""}">${state}</span></span>
      <span class="node-note">${escape(nodeNotes.has(nodeKey(n))?n.name:gpus[0]?.name ?? n.name)}</span>
      <span class="tiles">${gpus.map(g=>`<span title="${escape(deviceLabel(g))} · ${percent(g.util)}"><small>${escape(deviceLabel(g))}</small><b>${percent(g.util)}</b></span>`).join("")}</span>
    </button><button class="node-edit" type="button" data-note="${i}" aria-label="编辑 ${escape(nodeKey(n))} 的备注">备注</button></div>`;
  }).join("");
  renderOverview(nodes);
  const connection=$("connection");
  const ready=nodes.length>0 && available.length===nodes.length && !scope.disconnected;
  connection.textContent=ready?"采集中":scope.disconnected?"连接重试中":available.length?"部分在线":"等待采集";
  connection.classList.toggle("ready",ready);
  const times=available.map(n=>n.latest.time);
  count($("updated"), times.length?"最近采样 "+when(Math.min(...times)):"");
}

function renderCards() {
  const node=current();
  const settings=node?.config;
  const gpus=node?.latest?.gpus ?? [];
  const eligible=gpus.filter(g=>g.controllable!==false);
  const hasFreshData=node && !stale(node) && !!settings && node.latest.controllable && !scope.disconnected;
  count($("control-source"), node?.latest && !node.latest.controllable ? "只读采集" : "");
  const signature=JSON.stringify(gpus.map(g=>[g.uuid,g.index,g.label]));
  if ($("cards").dataset.signature!==signature) {
    $("cards").dataset.signature=signature;
    $("cards").innerHTML=gpus.map(g=>`<button type="button" data-card="${escape(g.uuid)}" aria-pressed="false">${escape(deviceLabel(g))}</button>`).join("");
    $("trend-card").innerHTML=gpus.map(g=>`<option value="${g.index}">${escape(deviceLabel(g))}</option>`).join("");
    if (!gpus.some(g=>g.index===scope.card)) scope.card=gpus[0]?.index??0;
    $("trend-card").value=scope.card;
  }
  const selected=chosen();
  for (const uuid of selected) if (!gpus.some(g=>g.uuid===uuid)) selected.delete(uuid);
  count($("node-label"),node?.name ?? "");
  $("gpu-table").tBodies[0].innerHTML=gpus.map(g=>{
    const desired=settings?.cards?.[g.uuid] ?? g.requested;
    const pending=desired?.enabled!==g.requested?.enabled || desired?.target!==g.requested?.target;
    const status=pending?"待生效":{off:"已关闭",running:"占卡中",yielding:"任务优先",recovery_hold:"待重新开启"}[g.worker_state];
    return `<tr><td><button type="button" class="gpu-pick" data-gpu="${g.index}" aria-pressed="${g.index===scope.card}">${escape(deviceLabel(g))}</button></td>
    <td>${meter(g.util,percent(g.util))}</td><td>${meter(ratio(g.used,g.total),memory(g.used)+" / "+memory(g.total))}</td>
    <td>${num(g.temp)} °C</td><td>${num(g.power)} W</td><td><span class="pill-status ${["off","recovery_hold"].includes(g.worker_state)?"off":""}"
    ${g.worker_state==="recovery_hold"?'title="监控已恢复，可手动重新开启占卡"':""}>${escape(status)}</span> ${desired?.enabled?percent(desired.target):""}</td></tr>`;
  }).join("");
  for (const btn of $("cards").querySelectorAll("button")) {
    const id=btn.dataset.card;
    const desired=settings?.cards?.[id];
    const allowed=eligible.some(g=>g.uuid===id);
    btn.title=allowed ? (desired?.enabled ? "目标 " + desired.target + "%" : "已关闭") : "该卡未启用控制";
    btn.disabled=!hasFreshData || scope.posting || !allowed;
  }
  $("apply").disabled=$("disable").disabled=!hasFreshData || scope.posting || !eligible.some(g=>selected.has(g.uuid));
  $("select-all").disabled=!hasFreshData || scope.posting;
  paintCardSelection();
  count($("select-all"),eligible.length>0 && eligible.every(g=>selected.has(g.uuid))?"清空选择":"全选");
}

function renderProcesses() {
  const node=current();
  const source=scope.history ? [...(node?.events ?? [])].reverse() : node?.latest?.processes ?? [];
  const rows=source.filter(p=>!$("category").value || $("category").value===p.category);
  $("proc-table").tBodies[0].innerHTML=rows.length?rows.map(p=>`<tr>
    <td>${escape(p.label ?? "GPU " + p.gpu)}</td><td class="mono">${escape(p.pid)}</td><td class="mono">${escape(p.local_pid)}</td>
    <td><span class="role ${classFor(p.category)}">${escape(p.category)}</span></td><td>${memory(p.memory)}</td>
    <td>${scope.history ? escape(p.action)+" · "+when(p.time,dated) : when(p.first_seen,dated)}</td>
    <td>${escape(p.evidence)} <small>${p.confidence ? "· "+escape(p.confidence) : ""}</small>
        ${p.command || p.user ? `<div class="detail">${escape(p.command ?? "")} ${p.user?"· "+escape(p.user):""}</div>` : ""}</td></tr>`).join("")
    : `<tr><td class="empty-cell" colspan="7">${node?.latest ? "暂无符合条件的进程记录" : "正在等待节点数据"}</td></tr>`;
  const candidates=node?.latest?.candidates ?? [];
  $("candidates").hidden=!candidates.length || scope.history;
  $("candidates").querySelector("ul").innerHTML=candidates.map(p=>`<li>${escape(p.category)} · 容器 PID ${escape(p.local_pid)} · GPU ${escape(p.gpus.join(", "))} · ${escape(p.evidence)}</li>`).join("");
}

function draw() {
  const node=current();
  const metrics=node?.metrics ?? [];
  const device=node?.latest?.gpus.find(g=>g.index===scope.card);
  const canvas=$("line"), ctx=canvas.getContext("2d"), box=canvas.getBoundingClientRect();
  if (!box.width) return;
  const dpr=Math.min(2,devicePixelRatio||1), width=box.width, height=230;
  canvas.width=Math.round(width*dpr); canvas.height=height*dpr;
  ctx.scale(dpr,dpr);
  const left=45,right=width-17,top=15,bottom=height-30;
  const end=metrics[metrics.length-1]?.t, start=end && Math.min(end-60,metrics[0].t);
  $("trend-empty").hidden=metrics.length>1;
  count($("trend-label"),end ? `${when(start)}–${when(end)} · ${metrics.length} 个样本` : "");
  const theme=getComputedStyle(document.documentElement);
  const color=name=>theme.getPropertyValue(name).trim();
  ctx.font="11px system-ui";
  for (const fraction of [0,.25,.5,.75,1]) {
    const y=bottom-(bottom-top)*fraction;
    ctx.strokeStyle=color("--chart-grid"); ctx.beginPath(); ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();
    ctx.fillStyle=color("--chart-text");ctx.fillText((100*fraction)+"%",6,y+4);
  }
  if (!end) return;
  for (let i=0;i<4;i++) {
    const t=start+(end-start)*i/3, x=left+(right-left)*i/3;
    ctx.textAlign=i===0?"left":i===3?"right":"center";ctx.fillText(when(t),x,height-9);
  }
  const lines=[[color("--cyan"),v=>v?.[0]],[color("--chart-memory"),v=>ratio(v?.[1],v?.[2])]];
  for(const [stroke,value] of lines) {
    let prev=null;
    ctx.beginPath();ctx.strokeStyle=stroke;ctx.lineWidth=2;
    for (const sample of metrics) {
      const metric=value((device && sample.g[device.uuid]) ?? sample.g[scope.card]);
      if (!Number.isFinite(metric)) {prev=null;continue;}
      const x=left+(right-left)*(sample.t-start)/(end-start);
      const y=bottom-(bottom-top)*Math.min(100,Math.max(0,metric))/100;
      if (prev!==null && sample.t-prev<30) ctx.lineTo(x,y);
      else ctx.moveTo(x,y);
      prev=sample.t;
    }
    ctx.stroke();
  }
}

async function hydrate() {
  if (scope.reading) return;
  scope.reading=true;
  try {
    const res=await fetch("/api/status",{cache:"no-store",signal:AbortSignal.timeout(7000)});
    if (!res.ok) throw Error("连接失败");
    scope.snapshot=await res.json();
    scope.disconnected=false;
  } catch { scope.disconnected=true; }
  finally { scope.reading=false; }
  const wanted=scope.pending;
  if (wanted && scope.snapshot.nodes.find(n=>n.name===wanted.node)?.latest?.revision>=wanted.revision) {
    count($("notice"),"已生效 · " + wanted.labels.join("、"));
    scope.pending=null;
  }
  renderNodes(); renderCards(); renderProcesses(); draw();
}

async function sendSettings(enabled) {
  const node=current(), selected=chosen(), target=Number($("target").value);
  const cards=(node?.latest?.gpus ?? []).filter(g=>g.controllable!==false && selected.has(g.uuid)).map(g=>g.uuid);
  if (!node?.config || !cards.length || scope.posting) return;
  if (!Number.isInteger(target) || target<1 || target>100) {
    count($("notice"),"请输入 1–100 的整数"); return;
  }
  const payload={node:node.name,cards,enabled,target,revision:node.config.revision};
  scope.posting=true;renderCards();
  count($("notice"),"正在提交...");
  try {
    const res=await fetch("/api/settings",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)});
    const result=await res.json();
    if (!res.ok || !result.ok) throw Error(result.error || "更新失败");
    scope.pending={node:node.name,revision:result.config.revision,
      labels:node.latest.gpus.filter(g=>cards.includes(g.uuid)).map(deviceLabel)};
    count($("notice"),"已保存，等待节点应用");
  } catch(err) { count($("notice"),err.message); }
  finally { scope.posting=false; await hydrate(); renderCards(); }
}


const cardGrid=$("cards"), marquee=$("card-marquee");
const cardButtons=()=>Array.from(cardGrid.querySelectorAll("[data-card]"));
const cardId=button=>button.dataset.card;
let cardDrag=null, ignoreCardClick=false;

function paintCardSelection() {
  const selected=cardDrag?.preview ?? chosen();
  for (const button of cardButtons()) button.setAttribute("aria-pressed",selected.has(cardId(button)));
}

function updateCardDrag(event) {
  const drag=cardDrag;
  if (!drag || drag.id!==event.pointerId) return;
  const x=event.clientX, y=event.clientY;
  if (!drag.preview) {
    if (Math.hypot(x-drag.x,y-drag.y)<5) return;
    cardGrid.setPointerCapture(drag.id);
  }
  const bounds=cardGrid.getBoundingClientRect();
  const clamp=(value,min,max)=>Math.min(max,Math.max(min,value));
  const left=clamp(Math.min(drag.x,x),bounds.left,bounds.right);
  const right=clamp(Math.max(drag.x,x),bounds.left,bounds.right);
  const top=clamp(Math.min(drag.y,y),bounds.top,bounds.bottom);
  const bottom=clamp(Math.max(drag.y,y),bounds.top,bounds.bottom);
  drag.preview=new Set();
  for (const button of cardButtons()) {
    const rect=button.getBoundingClientRect();
    if (!button.disabled && rect.right>left && rect.left<right && rect.bottom>top && rect.top<bottom) {
      drag.preview.add(cardId(button));
    }
  }
  Object.assign(marquee.style,{
    left:left-bounds.left+"px",top:top-bounds.top+"px",
    width:right-left+"px",height:bottom-top+"px",
  });
  marquee.hidden=false;
  paintCardSelection();
}

function finishCardDrag(commit=false) {
  if (!cardDrag) return;
  const {id,preview,selected}=cardDrag;
  cardDrag=null;
  marquee.hidden=true;
  if (!preview) return;
  if (cardGrid.hasPointerCapture(id)) cardGrid.releasePointerCapture(id);
  if (commit && selected===chosen() && !$("select-all").disabled) {
    selected.clear();
    for (const card of preview) selected.add(card);
  }
  renderCards();
}

cardGrid.addEventListener("pointerdown",event=>{
  if (!event.isPrimary || event.button!==0 || event.pointerType==="touch" || $("select-all").disabled) return;
  ignoreCardClick=false;
  cardDrag={id:event.pointerId,x:event.clientX,y:event.clientY,selected:chosen(),preview:null};
});
window.addEventListener("pointermove",updateCardDrag);
window.addEventListener("pointerup",event=>{
  if (cardDrag?.id!==event.pointerId) return;
  if (cardDrag.preview) updateCardDrag(event);
  ignoreCardClick=!!cardDrag.preview;
  finishCardDrag(true);
});
for (const name of ["pointercancel","lostpointercapture"]) {
  cardGrid.addEventListener(name,event=>{
    if (cardDrag?.id===event.pointerId) finishCardDrag();
  });
}
window.addEventListener("blur",()=>finishCardDrag());
window.addEventListener("keydown",event=>{
  if (event.key==="Escape" && cardDrag?.preview) {
    ignoreCardClick=true;
    finishCardDrag();
  }
});

$("nodes").addEventListener("click",e=>{
  const edit=e.target.closest("[data-note]");
  if (edit) {
    const node=scope.snapshot.nodes[Number(edit.dataset.note)];
    if (node) editNote(node);
    return;
  }
  const button=e.target.closest("[data-node]");
  if (!button) return;
  finishCardDrag();
  scope.node=Number(button.dataset.node);renderNodes();renderCards();renderProcesses();draw();
});
cardGrid.addEventListener("click",e=>{
  if (ignoreCardClick && e.detail) {
    ignoreCardClick=false;
    return;
  }
  const button=e.target.closest("[data-card]");
  if (!button || button.disabled) return;
  const uuid=button.dataset.card, selected=chosen();
  selected.has(uuid)?selected.delete(uuid):selected.add(uuid);renderCards();
});
$("select-all").onclick=()=>{
  const selected=chosen(), gpus=(current()?.latest?.gpus ?? []).filter(g=>g.controllable!==false);
  if (gpus.every(g=>selected.has(g.uuid))) selected.clear(); else gpus.forEach(g=>selected.add(g.uuid));
  renderCards();
};
$("gpu-table").addEventListener("click",e=>{
  const button=e.target.closest("[data-gpu]");
  if (button) {scope.card=Number(button.dataset.gpu);$("trend-card").value=scope.card;renderCards();draw();}
});
$("control").onsubmit=e=>{e.preventDefault();sendSettings(true);};
$("disable").onclick=()=>sendSettings(false);
$("category").onchange=renderProcesses;
$("trend-card").onchange=e=>{scope.card=Number(e.target.value);renderCards();draw();};
for(const [id,history] of [["tab-live",false],["tab-history",true]]) {
  $(id).onclick=()=>{ scope.history=history;$("tab-live").setAttribute("aria-pressed",!history);$("tab-history").setAttribute("aria-pressed",history);renderProcesses(); };
}
window.addEventListener("resize",draw);

const systemTheme=matchMedia("(prefers-color-scheme: dark)");
const themeControl=$("theme");
const themeKey="gpu-dashboard-theme";
function applyTheme() {
  const choice=themeControl.value;
  document.documentElement.dataset.theme=choice==="system" ? (systemTheme.matches ? "dark" : "light") : choice;
  draw();
}
try {
  const saved=localStorage.getItem(themeKey);
  if (saved==="light" || saved==="dark") themeControl.value=saved;
} catch {}
themeControl.addEventListener("change",()=>{
  try {
    if (themeControl.value==="system") localStorage.removeItem(themeKey);
    else localStorage.setItem(themeKey,themeControl.value);
  } catch {}
  applyTheme();
});
systemTheme.addEventListener("change",()=>{
  if (themeControl.value==="system") applyTheme();
});
applyTheme();

hydrate(); setInterval(hydrate,5000);
setInterval(()=>count($("clock"),when(Date.now()/1000)),1000);
