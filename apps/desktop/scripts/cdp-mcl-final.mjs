#!/usr/bin/env node
/**
 * Final acceptance driver for math-copy-latex (run against the e2e harness).
 * Asserts the FULL contract in one pass; prints PASS/FAIL lines; exits 1 on any fail.
 * Waits patiently for boot (page target + shell + plugin registration).
 */
import WebSocket from 'ws'

const sleep = ms => new Promise(r => setTimeout(r, ms))
const results = []
const check = (name, ok, detail = '') => {
  results.push({ name, ok })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`)
}

async function findPageTarget() {
  for (let i = 0; i < 90; i++) {
    try {
      const list = await (await fetch('http://127.0.0.1:9333/json/list')).json()
      const page = list.find(t => t.type === 'page' && t.url.includes('5174'))
      if (page) return page
    } catch {}
    await sleep(2000)
  }
  throw new Error('no page target on 9333 after 180s')
}

const page = await findPageTarget()
const ws = new WebSocket(page.webSocketDebuggerUrl)
await new Promise(r => ws.on('open', r))
let id = 0
const pending = new Map()
ws.on('message', raw => {
  const m = JSON.parse(raw.toString())
  if (m.id && pending.has(m.id)) {
    const { res, rej } = pending.get(m.id)
    pending.delete(m.id)
    m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result)
  }
})
const send = (method, params) => new Promise((res, rej) => {
  const mid = ++id
  pending.set(mid, { res, rej })
  ws.send(JSON.stringify({ id: mid, method, params }))
})
const evalJs = async expr => {
  const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })
  if (r.exceptionDetails) throw new Error('eval: ' + JSON.stringify(r.exceptionDetails).slice(0, 300))
  return r.result.value
}
const readClip = () => evalJs(`(window.hermesDesktop && window.hermesDesktop.readClipboard) ? window.hermesDesktop.readClipboard() : 'no-bridge'`)

// ── wait for shell + plugin registration ──────────────────────────────
let registered = false
for (let i = 0; i < 60; i++) {
  try {
    const s = await evalJs(`(async () => {
      if (!window.__HERMES_PLUGIN_SDK__) return 'sdk-down'
      try {
        const reg = await import('/src/contrib/registry.ts')
        return reg.registry.getArea('palette').some(c => String(c.id).includes('math-copy')) ? 'ready' : 'rows-pending'
      } catch { return 'rows-pending' }
    })()`)
    if (s === 'ready') { registered = true; break }
  } catch {}
  await sleep(2000)
}
check('plugin loads & contributes palette rows', registered)

// ── session + math prompt through the app's own gateway ───────────────
const submit = JSON.parse(await evalJs(`(async () => {
  const sdk = window.__HERMES_PLUGIN_SDK__
  try {
    const created = await sdk.host.request('session.create', { title: 'mcl-acceptance' })
    const sid = created && (created.session_id || created.id)
    await sdk.host.request('prompt.submit', { session_id: sid, text: 'Show the gaussian integral and mass-energy equivalence.' })
    return JSON.stringify({ ok: true, sid })
  } catch (e) { return JSON.stringify({ error: String(e && e.message || e) }) }
})()`))
check('session.create + prompt.submit', !!submit.ok, submit.error || submit.sid)

// ── KaTeX: the transcript shows the DRAFT view after session.create (normal
// app behavior) — open a completed session row through the real UI path. ──
let katexReady = false
for (let i = 0; i < 45; i++) {
  // If a session with rendered math is already open, done.
  const n = await evalJs(`document.querySelectorAll('.katex').length`)
  if (n >= 2) { katexReady = true; break }
  // Otherwise create a fresh turn and open it via the sidebar row (real UI).
  await evalJs(`(async () => {
    const sdk = window.__HERMES_PLUGIN_SDK__
    if (!sdk) return
    const created = await sdk.host.request('session.create', { title: 'mcl-acceptance-' + Date.now() })
    const sid = created.session_id || created.id
    await sdk.host.request('prompt.submit', { session_id: sid, text: 'gaussian please' })
  })()`)
  // wait for the turn to finish, then open the newest matching sidebar row
  await sleep(14000)
  await evalJs(`(async () => {
    const rows = Array.from(document.querySelectorAll('button, [role=button], a, [class*=session]'))
      .filter(el => (el.textContent || '').trim().startsWith('mcl-acceptance'))
    if (rows.length) rows[0].click()
  })()`)
  await sleep(3000)
  const n2 = await evalJs(`document.querySelectorAll('.katex').length`)
  if (n2 >= 2) { katexReady = true; break }
}
check('KaTeX formulas render in transcript', katexReady)
// retry loop may leave duplicated streamed turns (mock stream lacks finish_reason)
await evalJs(`window.getSelection().removeAllRanges()`)

// ── stylesheet injected ────────────────────────────────────────────────
check('plugin stylesheet injected', await evalJs(`!!document.getElementById('hermes-plugin-math-copy-latex-style')`))

// ── hover applies ──────────────────────────────────────────────────────
check('hover highlight applies', await evalJs(`(() => {
  const root = document.querySelectorAll('.katex')[0]
  root.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }))
  return root.classList.contains('hermes-mcl-hover')
})()`))

const clickFormula = i => evalJs(`(async () => {
  const root = document.querySelectorAll('.katex')[${i}]
  const r = root.getBoundingClientRect()
  const el = root.querySelector('.katex-html') || root
  const opts = { bubbles: true, cancelable: true, view: window, clientX: r.left + 2, clientY: r.top + Math.max(5, r.height / 2), button: 0 }
  el.dispatchEvent(new PointerEvent('pointerdown', opts))
  el.dispatchEvent(new MouseEvent('mousedown', opts))
  el.dispatchEvent(new MouseEvent('click', opts))
  await new Promise(rr => setTimeout(rr, 600))
  return true
})()`)

// ── click inline formula → $E = mc^2$ ──────────────────────────────────
await clickFormula(0)
await sleep(300)
check('click inline copies $E = mc^2$', JSON.stringify(await readClip()) === JSON.stringify('$E = mc^2$'), JSON.stringify(await readClip()))

// ── click integral → full LaTeX with delimiters ───────────────────────
await clickFormula(1)
await sleep(300)
check('click integral copies LaTeX', JSON.stringify(await readClip()) === JSON.stringify('$\\\\int_{-\\\\infty}^{\\\\infty} e^{-x^2}\\\\,dx = \\\\sqrt{\\\\pi}$'), JSON.stringify(await readClip()))

// ── selection guard ────────────────────────────────────────────────────
await evalJs(`(() => {
  const paras = Array.from(document.querySelectorAll('p'))
  const target = paras.find(p => p.textContent.includes('and display'))
  const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT)
  let node
  while ((node = walker.nextNode())) {
    if (node.textContent.trim().length > 3 && !node.parentElement.closest('.katex')) {
      const range = document.createRange()
      range.setStart(node, 0); range.setEnd(node, node.textContent.length)
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range)
      return true
    }
  }
  return false
})()`)
await clickFormula(2)
await sleep(200)
check('text selection blocks copy', JSON.stringify(await readClip()) === JSON.stringify('$\\\\int_{-\\\\infty}^{\\\\infty} e^{-x^2}\\\\,dx = \\\\sqrt{\\\\pi}$'))
await evalJs(`window.getSelection().removeAllRanges()`)

// ── palette toggle OFF: hover inert + click no-op ──────────────────────
await evalJs(`(async () => {
  const reg = await import('/src/contrib/registry.ts')
  reg.registry.getArea('palette').find(c => String(c.id) === 'math-copy-latex:toggle-click-to-copy').data.run()
})()`)
await sleep(150)
const detailOff = await evalJs(`(async () => (await import('/src/contrib/registry.ts')).registry.getArea('palette').find(c => String(c.id) === 'math-copy-latex:toggle-click-to-copy').data.detail())()`)
check('palette toggle flips to off', detailOff === 'off', detailOff)
check('hover inert while off', await evalJs(`(() => {
  const root = document.querySelectorAll('.katex')[0]
  root.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }))
  return !root.classList.contains('hermes-mcl-hover')
})()`))
await clickFormula(0)
await sleep(300)
check('click no-op while off', JSON.stringify(await readClip()) === JSON.stringify('$\\\\int_{-\\\\infty}^{\\\\infty} e^{-x^2}\\\\,dx = \\\\sqrt{\\\\pi}$'))

// ── palette toggle ON again → copy restored ────────────────────────────
await evalJs(`(async () => {
  const reg = await import('/src/contrib/registry.ts')
  reg.registry.getArea('palette').find(c => String(c.id) === 'math-copy-latex:toggle-click-to-copy').data.run()
})()`)
await sleep(150)
await clickFormula(0)
await sleep(300)
check('click restores after toggle on', JSON.stringify(await readClip()) === JSON.stringify('$E = mc^2$'), JSON.stringify(await readClip()))

// ── delimiter toggle → bare source ─────────────────────────────────────
await evalJs(`(async () => {
  const reg = await import('/src/contrib/registry.ts')
  reg.registry.getArea('palette').find(c => String(c.id) === 'math-copy-latex:toggle-delimiters').data.run()
})()`)
await sleep(150)
await clickFormula(0)
await sleep(300)
check('delimiter-off copies bare source', JSON.stringify(await readClip()) === JSON.stringify('E = mc^2'), JSON.stringify(await readClip()))

// ── restore defaults (delimiters ON, copy ON) ──────────────────────────
await evalJs(`(async () => {
  const reg = await import('/src/contrib/registry.ts')
  const d = reg.registry.getArea('palette').find(c => String(c.id) === 'math-copy-latex:toggle-delimiters')
  if (d.data.detail() !== 'with $…$') d.data.run()
})()`)
const final = await evalJs(`(async () => {
  const reg = await import('/src/contrib/registry.ts')
  return JSON.stringify({
    copy: reg.registry.getArea('palette').find(c => String(c.id) === 'math-copy-latex:toggle-click-to-copy').data.detail(),
    delims: reg.registry.getArea('palette').find(c => String(c.id) === 'math-copy-latex:toggle-delimiters').data.detail()
  })
})()`)
check('defaults restored (on / with $…$)', final === JSON.stringify({ copy: 'on', delims: 'with $…$' }), final)

ws.close()
const failed = results.filter(r => !r.ok)
console.log(`\\n${results.length - failed.length}/${results.length} checks passed`)
process.exit(failed.length ? 1 : 0)
