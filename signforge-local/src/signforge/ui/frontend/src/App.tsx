import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import {
    AlertCircle,
    Image as ImageIcon,
    Loader2,
    Sparkles,
    Wand2,
    Upload,
    X,
    Cpu,
    Activity,
    Database,
    ShieldCheck,
    ChevronRight,
    History,
    Settings2,
    Layers,
    MessageCircle,
    Send,
    User
} from 'lucide-react'

// Types
interface Adapter {
    name: string
    domain: string
    path: string
    recommended_weight: number
}

interface StatusResponse {
    id: string
    status: string
    progress: number
    total_steps: number
    image_url?: string
    error?: string
}

const RoyalCard = ({ children, title, icon: Icon, className = "" }: any) => (
    <motion.div
        whileHover={{ y: -5, scale: 1.01 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`glass-panel rounded-3xl p-6 ${className}`}
    >
        {title && (
            <div className="flex items-center gap-3 mb-6">
                <div className="p-2 rounded-xl bg-amber-500/5 border border-white/5">
                    <Icon className="w-5 h-5 text-amber-200/50" />
                </div>
                <h3 className="text-lg font-semibold gold-gradient-text tracking-wide">{title}</h3>
            </div>
        )}
        {children}
    </motion.div>
)

function App() {
    const [prompt, setPrompt] = useState('')
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState<StatusResponse | null>(null)
    const [adapters, setAdapters] = useState<Record<string, Adapter[]>>({})
    const [selectedAdapters, setSelectedAdapters] = useState<string[]>([])
    const [error, setError] = useState<string | null>(null)
    const [modelLoaded, setModelLoaded] = useState(false)
    const [modelLoading, setModelLoading] = useState(false)
    const [logoImage, setLogoImage] = useState<string | null>(null)
    const [backgroundImage, setBackgroundImage] = useState<string | null>(null)
    const [health, setHealth] = useState<any>(null)
    const [showMonitor, setShowMonitor] = useState(false)
    const [messages, setMessages] = useState<{ role: string, content: string }[]>([])
    const [chatInput, setChatInput] = useState('')
    const [isChatting, setIsChatting] = useState(false)
    const [showChat, setShowChat] = useState(false)
    const chatEndRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (showChat) {
            scrollToBottom()
        }
    }, [messages, showChat])

    const scrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }

    useEffect(() => {
        fetchAdapters()
        checkHealth()
        const interval = setInterval(checkHealth, 5000)
        return () => clearInterval(interval)
    }, [])

    const checkHealth = async () => {
        try {
            const res = await axios.get('/health')
            setHealth(res.data)
            setModelLoaded(res.data.model_loaded)
            setModelLoading(res.data.model_loading)
        } catch (err) {
            console.error('Health check failed', err)
        }
    }

    const fetchAdapters = async () => {
        try {
            const res = await axios.get('/adapters')
            const adapterData = res.data?.adapters || {}
            setAdapters(adapterData)
        } catch (err) {
            console.error('Failed to fetch adapters', err)
        }
    }

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>, type: 'logo' | 'background') => {
        const file = e.target.files?.[0]
        if (!file) return

        const reader = new FileReader()
        reader.onloadend = () => {
            if (type === 'logo') setLogoImage(reader.result as string)
            else setBackgroundImage(reader.result as string)
        }
        reader.readAsDataURL(file)
    }

    const handleGenerate = async () => {
        const finalPrompt = prompt || (logoImage ? "A luxury brand storefront showcasing this logo" : "A royal gold plated signage on a marble wall")

        setLoading(true)
        setError(null)
        setResult(null)

        try {
            const res = await axios.post('/generate', {
                prompt: finalPrompt,
                width: 1024,
                height: 768,
                steps: 20,
                adapters: selectedAdapters,
                adapter_weights: selectedAdapters.map(() => 1.0),
                logo_image_b64: logoImage,
                background_image_b64: backgroundImage
            })

            pollStatus(res.data.item_id)
        } catch (err: any) {
            setError(err.response?.data?.error || 'Forge failed to ignite. Check system link.')
            setLoading(false)
        }
    }

    const pollStatus = async (itemId: string) => {
        const interval = setInterval(async () => {
            try {
                const res = await axios.get<StatusResponse>(`/generate/${itemId}`)

                if (res.data.status === 'completed') {
                    clearInterval(interval)
                    const resultRes = await axios.get(`/generate/${itemId}/result`)
                    setResult({ ...res.data, image_url: resultRes.data.image_url })
                    setLoading(false)
                } else if (res.data.status === 'failed') {
                    clearInterval(interval)
                    setError(res.data.error || 'The forge cooled unexpectedly.')
                    setLoading(false)
                } else {
                    setResult(res.data)
                }
            } catch (err) {
                clearInterval(interval)
                setLoading(false)
                setError('Link to the royal forge severed.')
            }
        }, 1000)
    }

    const handleChat = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!chatInput.trim() || isChatting) return

        const userMsg = { role: 'user', content: chatInput }
        setMessages(prev => [...prev, userMsg])
        setChatInput('')
        setIsChatting(true)

        try {
            const res = await axios.post('/chat', {
                message: chatInput,
                history: messages
            })
            setMessages(prev => [...prev, { role: 'assistant', content: res.data.response }])
        } catch (err) {
            setMessages(prev => [...prev, { role: 'assistant', content: "My cognitive link failed. Please retry your inquiry." }])
        } finally {
            setIsChatting(false)
        }
    }

    return (
        <div className="min-h-screen relative">
            {/* Model Safety Net Overlay */}
            <AnimatePresence>
                {health && health.model_files_exist === false && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 z-[1000] bg-slate-950/95 backdrop-blur-2xl flex items-center justify-center p-6"
                    >
                        <motion.div
                            initial={{ scale: 0.9, y: 20 }}
                            animate={{ scale: 1, y: 0 }}
                            className="max-w-xl w-full glass-panel rounded-[40px] p-12 text-center border-amber-500/20 shadow-[0_0_50px_rgba(245,158,11,0.1)]"
                        >
                            <div className="w-20 h-20 rounded-3xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mx-auto mb-8">
                                <AlertCircle className="w-10 h-10 text-amber-500" />
                            </div>
                            <h2 className="text-3xl font-bold gold-gradient-text mb-4 tracking-tight uppercase">Imperial Forge Offline</h2>
                            <p className="text-slate-400 text-sm leading-relaxed mb-8">
                                The AI models are missing from your workstation. To ignite the forge, please ensure the models are downloaded.
                            </p>

                            <div className="bg-white/5 border border-white/10 rounded-2xl p-6 text-left mb-8">
                                <p className="text-[10px] font-bold text-amber-500 uppercase tracking-widest mb-3">Run this in your terminal:</p>
                                <code className="text-xs text-slate-300 block bg-black/40 p-4 rounded-xl border border-white/5 font-mono">
                                    python scripts/download_models.py
                                </code>
                            </div>

                            <p className="text-[10px] text-slate-600 uppercase tracking-[0.3em] font-bold">SignForge Local v1.0.0 — Diagnostic Report</p>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Ambient Background Elements */}
            <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
                <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] bg-amber-500/5 blur-[120px] rounded-full" />
                <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/5 blur-[120px] rounded-full" />
            </div>

            <div className="relative z-10 max-w-[1400px] mx-auto px-8 py-12">
                {/* Royal Header */}
                <header className="flex flex-col md:flex-row items-center justify-between gap-8 mb-16">
                    <motion.div
                        initial={{ opacity: 0, x: -30 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center gap-6"
                    >
                        <motion.div
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            className="relative w-16 h-16 bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl flex items-center justify-center shadow-2xl border border-white/5"
                        >
                            <Sparkles className="w-9 h-9 text-amber-200/80" />
                        </motion.div>
                        <div>
                            <h1 className="text-5xl font-bold tracking-tighter mb-1">
                                Sign<span className="gold-gradient-text">Forge</span>
                            </h1>
                            <div className="flex items-center gap-2 text-slate-400 font-medium">
                                <span className="w-2 h-2 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]" />
                                <p className="text-sm uppercase tracking-[0.2em]">Imperial Mockup Studio v2.0</p>
                            </div>
                        </div>
                    </motion.div>

                    <div className="flex items-center gap-4">
                        <AnimatePresence>
                            {modelLoading && (
                                <motion.div
                                    initial={{ opacity: 0, scale: 0.9, x: 20 }}
                                    animate={{ opacity: 1, scale: 1, x: 0 }}
                                    exit={{ opacity: 0, scale: 0.9, x: 20 }}
                                    className="flex items-center gap-3 bg-white/2 border border-white/5 px-6 py-3 rounded-2xl text-slate-400 text-xs font-bold tracking-widest"
                                >
                                    <Loader2 className="w-4 h-4 animate-spin text-amber-200/50" />
                                    SYNCING NEURAL CORE
                                </motion.div>
                            )}
                        </AnimatePresence>

                        <motion.button
                            whileHover={{ scale: 1.02, backgroundColor: "rgba(255,255,255,0.08)" }}
                            whileTap={{ scale: 0.98 }}
                            onClick={() => setShowMonitor(!showMonitor)}
                            className={`px-6 py-3 rounded-2xl border flex items-center gap-3 text-xs font-bold tracking-widest transition-all ${showMonitor ? 'bg-white/10 border-white/20 text-white shadow-lg' : 'bg-white/2 border-white/5 text-slate-500'}`}
                        >
                            <Activity className="w-4 h-4" /> MONITOR
                        </motion.button>
                    </div>
                </header>

                <AnimatePresence>
                    {showMonitor && health && (
                        <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="overflow-hidden mb-12"
                        >
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 p-1">
                                {[
                                    { label: 'Core Processor', val: health.device === 'cpu' ? 'System CPU' : health.device.toUpperCase(), sub: `DType: ${health.dtype}`, icon: Cpu },
                                    { label: 'Task Queue', val: `${health.queue_size} / ${health.queue_max}`, sub: 'Load Level', icon: History, progress: (health.queue_size / health.queue_max) * 100 },
                                    { label: 'Imperial VRAM', val: health.gpu_memory_gb ? `${health.gpu_used_gb.toFixed(1)} GB` : 'System Path', sub: health.gpu_memory_gb ? `Total: ${health.gpu_memory_gb.toFixed(1)} GB` : 'No Dedicated GPU', icon: Database },
                                    { label: 'Engine Status', val: 'LEGITIMATE', sub: 'Verified Session', icon: ShieldCheck, accent: true }
                                ].map((stat, i) => (
                                    <motion.div
                                        key={i}
                                        whileHover={{ y: -5, backgroundColor: "rgba(255,255,255,0.03)" }}
                                        className="glass-panel p-6 rounded-3xl relative overflow-hidden group transition-colors"
                                    >
                                        <div className="absolute right-[-10px] top-[-10px] scale-150 opacity-[0.02] group-hover:scale-[1.7] group-hover:opacity-[0.04] transition-all duration-700">
                                            <stat.icon className="w-24 h-24" />
                                        </div>
                                        <div className="flex items-center gap-3 mb-4">
                                            <div className={`p-2 rounded-lg ${stat.accent ? 'bg-blue-500/5' : 'bg-white/2'}`}>
                                                <stat.icon className={`w-4 h-4 ${stat.accent ? 'text-blue-400/50' : 'text-slate-600'}`} />
                                            </div>
                                            <p className="text-[10px] font-bold text-slate-600 uppercase tracking-widest">{stat.label}</p>
                                        </div>
                                        <p className="text-2xl font-bold tracking-tight mb-1 text-slate-300">{stat.val}</p>
                                        <p className="text-xs text-slate-600">{stat.sub}</p>
                                        {stat.progress !== undefined && (
                                            <div className="w-full bg-white/2 h-1 rounded-full mt-4 overflow-hidden">
                                                <div className="bg-amber-200/30 h-full" style={{ width: `${stat.progress}%` }} />
                                            </div>
                                        )}
                                    </motion.div>
                                ))}
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                    {/* Command Center */}
                    <div className="lg:col-span-4 space-y-8">
                        <RoyalCard title="Creation Commands" icon={Wand2}>
                            <div className="space-y-6">
                                <div className="space-y-3">
                                    <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Inspiration Prompt</label>
                                    <textarea
                                        value={prompt}
                                        onChange={(e) => setPrompt(e.target.value)}
                                        placeholder="E.g. A gold-etched boutique sign on ivory marble..."
                                        className="royal-input w-full h-32 rounded-2xl p-4 text-sm resize-none"
                                    />
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div className="space-y-3">
                                        <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Imperial Logo</label>
                                        <div className={`relative h-32 rounded-2xl border-2 border-dashed flex items-center justify-center overflow-hidden transition-all group ${logoImage ? 'border-amber-500/50 bg-amber-500/5' : 'border-white/5 hover:border-white/10 bg-white/2'}`}>
                                            {logoImage ? (
                                                <>
                                                    <img src={logoImage} className="w-full h-full object-contain p-4" alt="Logo" />
                                                    <button onClick={() => setLogoImage(null)} className="absolute top-2 right-2 p-1.5 bg-black/60 backdrop-blur-md rounded-xl hover:bg-black text-white transition-colors duration-200"><X className="w-3.5 h-3.5" /></button>
                                                </>
                                            ) : (
                                                <label className="cursor-pointer flex flex-col items-center gap-2 group-hover:scale-110 transition-transform duration-300">
                                                    <Upload className="w-5 h-5 text-slate-500 group-hover:text-amber-500 transition-colors" />
                                                    <span className="text-[10px] text-slate-500 font-bold tracking-widest">SVG/PNG</span>
                                                    <input type="file" className="hidden" accept="image/*" onChange={(e) => handleFileUpload(e, 'logo')} />
                                                </label>
                                            )}
                                        </div>
                                    </div>

                                    <div className="space-y-3">
                                        <label className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] ml-1">Facade/Wall</label>
                                        <div className={`relative h-32 rounded-2xl border-2 border-dashed flex items-center justify-center overflow-hidden transition-all group ${backgroundImage ? 'border-amber-500/50 bg-amber-500/5' : 'border-white/5 hover:border-white/10 bg-white/2'}`}>
                                            {backgroundImage ? (
                                                <>
                                                    <img src={backgroundImage} className="w-full h-full object-cover" alt="Wall" />
                                                    <button onClick={() => setBackgroundImage(null)} className="absolute top-2 right-2 p-1.5 bg-black/60 backdrop-blur-md rounded-xl hover:bg-black text-white transition-colors duration-200"><X className="w-3.5 h-3.5" /></button>
                                                </>
                                            ) : (
                                                <label className="cursor-pointer flex flex-col items-center gap-2 group-hover:scale-110 transition-transform duration-300">
                                                    <ImageIcon className="w-5 h-5 text-slate-500 group-hover:text-blue-500 transition-colors" />
                                                    <span className="text-[10px] text-slate-500 font-bold tracking-widest">UPLOAD</span>
                                                    <input type="file" className="hidden" accept="image/*" onChange={(e) => handleFileUpload(e, 'background')} />
                                                </label>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                <motion.button
                                    whileHover={{ scale: 1.01, boxShadow: "0 0 20px rgba(194,167,122,0.15)" }}
                                    whileTap={{ scale: 0.98 }}
                                    onClick={() => handleGenerate()}
                                    disabled={loading || !modelLoaded}
                                    className="royal-button w-full py-5 rounded-2xl flex items-center justify-center gap-3 group disabled:opacity-30 disabled:grayscale disabled:hover:scale-100"
                                >
                                    {loading ? (
                                        <Loader2 className="animate-spin" />
                                    ) : (
                                        <Sparkles className="w-5 h-5 opacity-50 group-hover:opacity-100 group-hover:rotate-12 transition-all" />
                                    )}
                                    <span className="uppercase tracking-[0.2em] text-xs font-black">
                                        {loading ? "Forging..." : !modelLoaded ? "Awaiting Core" : "Ignite Creation"}
                                    </span>
                                </motion.button>
                            </div>
                        </RoyalCard>

                        <RoyalCard title="LoRA Archive" icon={Layers}>
                            <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                                {Object.keys(adapters).length === 0 ? (
                                    <div className="p-4 rounded-xl bg-white/2 border border-white/5 text-center">
                                        <p className="text-xs text-slate-600 italic">No imperial scrolls found in library.</p>
                                    </div>
                                ) : (
                                    Object.entries(adapters).map(([domain, items]) => (
                                        <div key={domain} className="mb-4">
                                            <p className="text-[10px] font-bold text-amber-500 uppercase tracking-[0.2em] mb-3 ml-1">{domain}</p>
                                            <div className="space-y-1">
                                                {Array.isArray(items) && items.map(adapter => (
                                                    <button
                                                        key={adapter.name}
                                                        onClick={() => {
                                                            if (selectedAdapters.includes(adapter.name)) {
                                                                setSelectedAdapters(selectedAdapters.filter(a => a !== adapter.name))
                                                            } else {
                                                                setSelectedAdapters([...selectedAdapters, adapter.name])
                                                            }
                                                        }}
                                                        className={`w-full flex items-center justify-between p-3 rounded-xl transition-all duration-300 border ${selectedAdapters.includes(adapter.name) ? 'bg-amber-500/10 border-amber-500/30 text-amber-200' : 'bg-transparent border-transparent text-slate-500 hover:bg-white/5 hover:text-slate-300'}`}
                                                    >
                                                        <div className="flex items-center gap-3">
                                                            <div className={`w-1.5 h-1.5 rounded-full transition-all ${selectedAdapters.includes(adapter.name) ? 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.6)]' : 'bg-slate-700'}`} />
                                                            <span className="text-xs font-semibold">{adapter.name}</span>
                                                        </div>
                                                        {selectedAdapters.includes(adapter.name) && <ChevronRight className="w-3 h-3" />}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </RoyalCard>
                    </div>

                    {/* Imperial Canvas */}
                    <div className="lg:col-span-8">
                        <motion.div
                            layout
                            className="relative aspect-[4/3] w-full rounded-[40px] overflow-hidden glass-panel p-3 bg-white/[0.02]"
                        >
                            <div className="relative w-full h-full bg-slate-950 rounded-[32px] overflow-hidden flex items-center justify-center group shadow-inner">
                                <AnimatePresence mode="wait">
                                    {result?.image_url ? (
                                        <motion.img
                                            key="result"
                                            initial={{ opacity: 0, scale: 1.1 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            src={result.image_url}
                                            className="w-full h-full object-cover"
                                            alt="Imperial Output"
                                        />
                                    ) : loading ? (
                                        <motion.div
                                            key="loading"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            exit={{ opacity: 0 }}
                                            className="text-center"
                                        >
                                            <div className="relative mb-6">
                                                <div className="absolute inset-0 bg-amber-500/20 blur-3xl animate-pulse" />
                                                <Loader2 className="w-16 h-16 animate-spin text-amber-500 mx-auto relative z-10" />
                                            </div>
                                            <h3 className="text-xl font-bold gold-gradient-text tracking-widest mb-1">MAGNUM OPUS IN PROGRESS</h3>
                                            <p className="text-sm text-slate-500 font-medium">Mixing metallic particles and light photons...</p>
                                            {result?.progress !== undefined && (
                                                <div className="mt-8 max-w-[200px] mx-auto">
                                                    <div className="flex justify-between text-[10px] font-bold text-slate-500 mb-2 uppercase tracking-widest">
                                                        <span>Progress</span>
                                                        <span>{Math.round((result.progress / 20) * 100)}%</span>
                                                    </div>
                                                    <div className="w-full bg-white/5 h-1 rounded-full overflow-hidden">
                                                        <motion.div
                                                            className="h-full bg-gradient-to-r from-amber-600 to-amber-400"
                                                            animate={{ width: `${(result.progress / 20) * 100}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                        </motion.div>
                                    ) : (
                                        <motion.div
                                            key="idle"
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            className="text-center group"
                                        >
                                            <div className="w-32 h-32 mx-auto mb-8 rounded-full bg-white/[0.02] border border-white/[0.05] flex items-center justify-center group-hover:scale-110 transition-transform duration-700">
                                                <ImageIcon className="w-12 h-12 text-slate-800" />
                                            </div>
                                            <h3 className="text-2xl font-semibold text-slate-600 tracking-tight">Imperial Render Canvas</h3>
                                            <p className="text-slate-800 text-sm mt-2">Awaiting your master command</p>
                                        </motion.div>
                                    )}
                                </AnimatePresence>

                                {/* Canvas Overlays */}
                                <div className="absolute top-8 left-8 flex gap-3">
                                    <div className="px-4 py-2 bg-black/40 backdrop-blur-xl border border-white/5 rounded-xl text-[10px] font-bold text-slate-400 tracking-widest uppercase">
                                        MASTER OUTPUT
                                    </div>
                                    <div className="px-4 py-2 bg-black/40 backdrop-blur-xl border border-white/5 rounded-xl text-[10px] font-bold text-amber-500 tracking-widest uppercase">
                                        1024 x 768
                                    </div>
                                </div>

                                {error && (
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="absolute inset-x-0 bottom-0 p-8 bg-gradient-to-t from-slate-950/90 to-transparent"
                                    >
                                        <div className="flex items-center gap-3 text-slate-400">
                                            <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                            <p className="text-xs font-medium italic">{error}</p>
                                        </div>
                                    </motion.div>
                                )}
                            </div>
                        </motion.div>
                    </div>
                </div>

                <footer className="mt-24 pt-12 border-t border-white/5 text-center">
                    <p className="text-[10px] font-bold text-slate-700 uppercase tracking-[0.4em]">SignForge Local Studio &copy; 2026 — Crafted for Royalty</p>
                </footer>
            </div>

            {/* Imperial Chat Assistant */}
            <div className="fixed bottom-8 right-8 z-[100]">
                <AnimatePresence>
                    {showChat && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.8, y: 100, x: 50 }}
                            animate={{ opacity: 1, scale: 1, y: 0, x: 0 }}
                            exit={{ opacity: 0, scale: 0.8, y: 100, x: 50 }}
                            className="absolute bottom-20 right-0 w-96 h-[500px] glass-panel rounded-[32px] overflow-hidden flex flex-col shadow-[0_20px_50px_rgba(0,0,0,0.5)] border-white/10"
                        >
                            <div className="p-6 border-b border-white/5 bg-white/2 flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-full bg-amber-500/20 flex items-center justify-center">
                                        <Sparkles className="w-4 h-4 text-amber-500" />
                                    </div>
                                    <div>
                                        <h4 className="text-sm font-bold gold-gradient-text tracking-wide">FORGE ASSISTANT</h4>
                                        <p className="text-[9px] text-green-500 font-bold uppercase tracking-widest">Neural Link Active</p>
                                    </div>
                                </div>
                                <button onClick={() => setShowChat(false)} className="text-slate-500 hover:text-white transition-colors">
                                    <X className="w-4 h-4" />
                                </button>
                            </div>

                            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
                                {messages.length === 0 && (
                                    <div className="h-full flex flex-col items-center justify-center text-center px-4">
                                        <div className="w-12 h-12 rounded-full bg-white/2 border border-white/5 flex items-center justify-center mb-4">
                                            <MessageCircle className="w-6 h-6 text-slate-700" />
                                        </div>
                                        <p className="text-xs text-slate-500 font-medium">Greetings. I am the Imperial Assistant. Ask me anything about signage design or using the forge.</p>
                                    </div>
                                )}
                                {messages.map((msg, i) => (
                                    <motion.div
                                        initial={{ opacity: 0, x: msg.role === 'user' ? 20 : -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        key={i}
                                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                    >
                                        <div className={`max-w-[80%] p-4 rounded-2xl text-xs leading-relaxed ${msg.role === 'user'
                                            ? 'bg-amber-500/10 border border-amber-500/20 text-amber-100 rounded-tr-none'
                                            : 'bg-white/5 border border-white/10 text-slate-300 rounded-tl-none'
                                            }`}>
                                            {msg.content}
                                        </div>
                                    </motion.div>
                                ))}
                                {isChatting && (
                                    <div className="flex justify-start">
                                        <div className="bg-white/5 border border-white/10 p-4 rounded-2xl rounded-tl-none">
                                            <Loader2 className="w-3 h-3 animate-spin text-amber-500/50" />
                                        </div>
                                    </div>
                                )}
                                <div ref={chatEndRef} />
                            </div>

                            <form onSubmit={handleChat} className="p-4 bg-white/2 border-t border-white/5">
                                <div className="relative">
                                    <input
                                        type="text"
                                        value={chatInput}
                                        onChange={(e) => setChatInput(e.target.value)}
                                        placeholder="Command the assistant..."
                                        className="w-full bg-slate-950 border border-white/10 rounded-2xl py-3 pl-4 pr-12 text-xs focus:border-amber-500/50 outline-none transition-all"
                                    />
                                    <button
                                        type="submit"
                                        disabled={isChatting || !chatInput.trim()}
                                        className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-xl bg-amber-500 flex items-center justify-center text-black disabled:opacity-30 transition-opacity"
                                    >
                                        <Send className="w-3.5 h-3.5" />
                                    </button>
                                </div>
                            </form>
                        </motion.div>
                    )}
                </AnimatePresence>

                <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => setShowChat(!showChat)}
                    className={`w-14 h-14 rounded-full flex items-center justify-center shadow-2xl transition-colors ${showChat ? 'bg-white/10 text-white' : 'bg-amber-500 text-black shadow-[0_0_20px_rgba(245,158,11,0.4)]'}`}
                >
                    {showChat ? <X className="w-6 h-6" /> : <MessageCircle className="w-6 h-6" />}
                </motion.button>
            </div>
        </div>
    )
}

export default App
