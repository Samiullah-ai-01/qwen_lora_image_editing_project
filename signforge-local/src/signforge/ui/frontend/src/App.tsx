import { useState, useEffect } from 'react'
import axios from 'axios'
import { AlertCircle, Image as ImageIcon, Loader2, Sparkles, Wand2 } from 'lucide-react'

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

function App() {
    const [prompt, setPrompt] = useState('')
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState<StatusResponse | null>(null)
    const [adapters, setAdapters] = useState<Record<string, Adapter[]>>({})
    const [selectedAdapters, setSelectedAdapters] = useState<string[]>([])
    const [error, setError] = useState<string | null>(null)

    const [modelLoaded, setModelLoaded] = useState(false)

    useEffect(() => {
        fetchAdapters()
        checkHealth()
        const interval = setInterval(checkHealth, 5000)
        return () => clearInterval(interval)
    }, [])

    const checkHealth = async () => {
        try {
            const res = await axios.get('/health')
            setModelLoaded(res.data.model_loaded)
        } catch (err) {
            console.error('Health check failed', err)
        }
    }

    const fetchAdapters = async () => {
        try {
            const res = await axios.get('/adapters')
            // Safe access to nested adapters object
            const adapterData = res.data?.adapters || {}
            setAdapters(adapterData)
        } catch (err) {
            console.error('Failed to fetch adapters', err)
        }
    }

    const handleGenerate = async (e?: React.FormEvent) => {
        if (e) e.preventDefault()
        if (!prompt && !e) setPrompt("A futuristic neon sign for SignForge")

        const finalPrompt = prompt || "A futuristic neon sign for SignForge"

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
                adapter_weights: selectedAdapters.map(() => 1.0)
            })

            pollStatus(res.data.item_id)
        } catch (err: any) {
            setError(err.response?.data?.error || 'Generation failed to start. Is the server running?')
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
                    setError(res.data.error || 'Generation failed during processing')
                    setLoading(false)
                } else {
                    setResult(res.data)
                }
            } catch (err) {
                clearInterval(interval)
                setLoading(false)
                setError('Lost contact with server.')
            }
        }, 1000)
    }

    return (
        <div className="min-h-screen bg-[#020617] text-slate-50 font-sans">
            <div className="max-w-7xl mx-auto px-6 py-12">

                {/* Header */}
                <header className="flex items-center justify-between mb-12 pb-8 border-b border-slate-800">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <Sparkles className="w-8 h-8 text-blue-500" />
                            <h1 className="text-3xl font-bold">SignForge <span className="text-blue-500">Local</span></h1>
                        </div>
                        <p className="text-slate-400">High-fidelity signage mockup generator</p>
                    </div>

                    {!modelLoaded && (
                        <div className="flex items-center gap-4 bg-blue-500/10 border border-blue-500/20 px-4 py-2 rounded-xl text-blue-400 text-sm animate-pulse">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>Initializing Base Model (approx. 12GB download)...</span>
                        </div>
                    )}
                </header>

                <main className="grid grid-cols-1 lg:grid-cols-12 gap-12">
                    {/* Controls */}
                    <div className="lg:col-span-4 space-y-6">
                        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                                <Wand2 className="w-4 h-4" /> Studio Controls
                            </h3>

                            <div className="space-y-4">
                                <textarea
                                    value={prompt}
                                    onChange={(e) => setPrompt(e.target.value)}
                                    placeholder="Describe your signage vision..."
                                    className="w-full h-32 bg-slate-950 border border-slate-800 rounded-xl p-4 focus:ring-1 focus:ring-blue-500 outline-none transition-all text-sm"
                                />

                                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800">
                                    <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">Available LoRAs</p>
                                    <div className="space-y-1">
                                        {Object.keys(adapters).length === 0 ? (
                                            <p className="text-xs text-slate-600 italic">No LoRAs found in models/loras/</p>
                                        ) : (
                                            Object.entries(adapters).map(([domain, items]) => (
                                                Array.isArray(items) && items.map(adapter => (
                                                    <div key={adapter.name} className="text-xs py-1 text-slate-400">● {adapter.name}</div>
                                                ))
                                            ))
                                        )}
                                    </div>
                                </div>

                                <button
                                    onClick={() => handleGenerate()}
                                    disabled={loading}
                                    className="w-full py-4 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 text-white font-bold rounded-xl flex items-center justify-center gap-2"
                                >
                                    {loading ? <Loader2 className="animate-spin" /> : <Sparkles className="w-4 h-4" />}
                                    {loading ? "Processing..." : "Generate Mockup"}
                                </button>
                            </div>
                        </div>

                        {error && (
                            <div className="p-4 bg-red-500/10 border border-red-500/20 text-red-500 rounded-xl text-sm italic">
                                {error}
                            </div>
                        )}
                    </div>

                    {/* Canvas */}
                    <div className="lg:col-span-8">
                        <div className="aspect-video bg-slate-900 flex items-center justify-center rounded-3xl border border-slate-800 shadow-2xl relative overflow-hidden group">
                            {result?.image_url ? (
                                <img src={result.image_url} className="w-full h-full object-cover" alt="Output" />
                            ) : loading ? (
                                <div className="text-center">
                                    <Loader2 className="w-12 h-12 animate-spin text-blue-500 mx-auto mb-4" />
                                    <p className="text-slate-400">Forging mock image...</p>
                                </div>
                            ) : (
                                <div className="text-center opacity-30 group-hover:opacity-50 transition-opacity">
                                    <ImageIcon className="w-24 h-24 mx-auto mb-4" />
                                    <p className="text-xl font-medium">Render Canvas</p>
                                </div>
                            )}

                            <div className="absolute bottom-6 right-6 px-3 py-1 bg-black/50 backdrop-blur-md rounded-lg text-[10px] font-mono text-slate-500">
                                RESOLUTION: 1024x768
                            </div>
                        </div>
                    </div>
                </main>
            </div>
        </div>
    )
}

export default App
