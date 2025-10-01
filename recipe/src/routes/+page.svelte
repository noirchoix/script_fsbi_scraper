<script lang="ts">
    import { fly, fade } from 'svelte/transition';
    import { quintOut } from 'svelte/easing';
    import { onDestroy, tick } from 'svelte';
    import { startScrapeJob, getJobStatus, exportJobData } from '$lib/api';
    import type { ScrapeConfig, JobResponse } from '$lib/types'; 

    // --- CONFIGURATION ---
    const POLLING_RATE_MS = 3000; // Check job status every 3 seconds
    
    // --- STATE MANAGEMENT ---
    
    // Form Inputs
    let discoverTerms: string = "apple, cinnamon, coffee";
    let maxCompounds: number = 500;
    let delay: number = 0.25;

    // Job Tracking & Polling State
    let currentJob: JobResponse | null = null; // Holds the last polled job status
    let isSubmitting: boolean = false;
    let error: string | null = null;

    // Timer & Status Text
    let jobMessage: string = "Ready to start scraping.";
    let timeRemaining: number | null = null; // In seconds
    let timerInterval: number | null = null;
    let pollingInterval: number | null = null;
    
    // Pop-up/Modal State
    let showCompletionModal = false; // Controls the modal visibility

    // --- REACTIVE ESTIMATION ---
    
    /**
     * Calculates the estimated wait time (in seconds) based on form inputs.
     * This estimation is used for the initial countdown start.
     */
    $: estimatedWaitTimeSeconds = (() => {
        const compounds = Math.max(1, maxCompounds);
        const delayFactor = Math.max(0.01, delay);
        // Using the backend formula: max_compounds * delay * 2.5 (as defined in main.py heuristic)
        const baseEstimate = compounds * delayFactor * 2.5; 
        const safetyMargin = 30; 
        return Math.ceil(baseEstimate + safetyMargin);
    })();


    // --- UTILITY FUNCTIONS ---

    /**
     * Formats seconds into a M:SS string.
     */
    function formatTime(seconds: number): string {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }

    /**
     * Starts the countdown timer from the estimated time.
     */
    function startCountdown() {
        if (!currentJob || !currentJob.estimated_time) return;

        // Start from the backend's estimated time
        timeRemaining = Math.ceil(currentJob.estimated_time);

        // Clear any existing timer just in case
        stopCountdown(); 
        
        timerInterval = window.setInterval(() => {
            if (timeRemaining && timeRemaining > 0) {
                timeRemaining--;
            } else if (timerInterval) {
                // Time's up! Stop the countdown, but keep polling.
                stopCountdown();
            }
        }, 1000);
    }

    /**
     * Stops the countdown timer.
     */
    function stopCountdown() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    // --- POLLING LOGIC ---

    function startPolling(jobId: number) {
        if (pollingInterval) stopPolling();

        pollingInterval = window.setInterval(async () => {
            try {
                const updatedJob = await getJobStatus(jobId);
                currentJob = updatedJob; // This triggers Svelte's reactivity to update the UI
                
                // Update dynamic message
                jobMessage = updatedJob.last_log_message;

                if (updatedJob.status === 'COMPLETED' || updatedJob.status === 'ERROR') {
                    stopPolling();
                    stopCountdown();
                    
                    if (updatedJob.status === 'COMPLETED') {
                        jobMessage = `Job #${updatedJob.id} Complete! Found ${updatedJob.total_foods_scraped} foods.`;
                        showCompletionModal = true;
                    } else if (updatedJob.status === 'ERROR') {
                        jobMessage = `Job Failed: ${updatedJob.last_log_message}`;
                        error = updatedJob.last_log_message;
                    }
                }
            } catch (e: any) {
                console.error(`Polling error for Job #${jobId}:`, e.message);
                // The job might have been deleted, or network failed.
                stopPolling();
                stopCountdown();
                error = `Lost connection to job #${jobId}. Please check the server logs.`;
                currentJob = null;
            }
        }, POLLING_RATE_MS);
    }

    function stopPolling() {
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    }

    /**
     * Handles the file export process.
     */
    async function handleExport() {
        if (!currentJob || currentJob.status !== 'COMPLETED') return;
        
        jobMessage = "Preparing CSV export...";
        error = null;
        
        try {
            // The exportJobData function should trigger a browser download 
            // if the backend returns a FileResponse.
            await exportJobData(currentJob.id);
            jobMessage = "Export complete. Check your downloads folder.";
        } catch (e: any) {
            jobMessage = "Export failed.";
            error = e.message;
        }
    }

    // --- SUBMISSION HANDLER ---

    async function handleScrape() {
        if (isSubmitting || (currentJob && currentJob.status === 'RUNNING')) return;

        // Cleanup previous job state
        stopPolling();
        stopCountdown();
        currentJob = null; 
        error = null;

        // Simple form validation check (already handled by 'required' attributes, but good for TS)
        if (!discoverTerms.trim() || maxCompounds < 1 || delay < 0) {
            error = "Please fill out all fields correctly.";
            return;
        }

        isSubmitting = true;
        jobMessage = "Submitting job request to FastAPI...";

        const config: ScrapeConfig = {
            discover_terms: discoverTerms, 
            max_compounds: maxCompounds,
            delay: delay,
        };

        try {
            const job = await startScrapeJob(config);
            currentJob = job;
            
            // 1. Start live monitoring
            startCountdown();
            startPolling(job.id);
            jobMessage = `Job #${job.id} Accepted: Starting background process...`;

        } catch (e) {
            console.error("Scrape initiation failed:", e);
            if (e instanceof Error) {
                error = e.message;
            } else {
                error = "An unexpected error occurred during job submission.";
            }
            jobMessage = "Error starting job.";
        } finally {
            isSubmitting = false;
        }
    }

    // Cleanup intervals when the component is destroyed (Svelte best practice)
    onDestroy(() => {
        stopPolling();
        stopCountdown();
    });
</script>

<div class="bg-white text-gray-800 dark:bg-neutral-950 relative isolate antialiased dark:text-neutral-100 min-h-screen">
    <header class="mx-auto items-center px-8 pt-10 text-sm relative z-10 flex max-w-7xl">
        <p class="font-semibold tracking-wider text-indigo-600 dark:text-indigo-400">FSBI-DB Scraper</p>
    </header>

    <main class="mx-auto px-8 relative z-20 max-w-7xl">
        <div class="lg:grid-cols-2 lg:py-24 items-start py-16 grid gap-12">
            <div class="space-y-8" transition:fly={{ y: 50, duration: 500, easing: quintOut }}>
                <div class="space-y-6">
                    <p class="text-5xl font-extrabold leading-tight lg:text-7xl text-gray-900 dark:text-neutral-50">
                        Flavor Database Scraper 🧪
                    </p>
                    <p class="text-lg text-gray-700 dark:text-neutral-300 max-w-xl">
                        Configure and run the FSBI-DB scraper to securely build your flavor and molecule datasets in the background. Jobs persist on the server.
                    </p>
                </div>

                <div class="space-y-4 pt-4 min-h-[50px]">
                    {#if currentJob}
                        <div transition:fly|local={{ y: -10, duration: 300 }} class="p-4 rounded-xl space-y-2 border 
                            {currentJob.status === 'COMPLETED' ? 'bg-green-900/10 border-green-500 text-green-300' :
                             currentJob.status === 'ERROR' ? 'bg-red-900/10 border-red-500 text-red-300' :
                             'bg-indigo-900/10 border-indigo-500 text-indigo-300'}">
                            
                            <p class="text-xs font-semibold uppercase">Job #{currentJob.id} Status</p>
                            <p class="text-lg font-bold">{jobMessage}</p>

                            {#if currentJob.status === 'RUNNING' || currentJob.status === 'PENDING'}
                                <div class="flex items-center justify-between text-sm text-gray-400 pt-2 border-t border-indigo-900/50">
                                    <p>Est. Remaining: <span class="font-mono text-indigo-400">
                                        {timeRemaining !== null ? formatTime(timeRemaining) : (currentJob.estimated_time ? formatTime(Math.ceil(currentJob.estimated_time)) : '...') }
                                    </span></p>
                                    <p>Progress: <span class="text-indigo-400">{currentJob.total_foods_scraped}</span> Foods Scraped</p>
                                </div>
                            {/if}
                        </div>
                    {/if}
                    
                    {#if error && (!currentJob || currentJob.status !== 'ERROR')}
                        <div transition:fly|local={{ y: -10, duration: 300 }} class="text-red-600 font-medium p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-900 rounded-lg">
                            ❌ Error: {error}
                        </div>
                    {/if}
                </div>
            </div>

            <div class="bg-neutral-50 dark:bg-neutral-900 p-8 rounded-xl shadow-2xl space-y-8">
                <h2 class="text-2xl font-semibold border-b pb-3 border-neutral-200 dark:border-neutral-800 text-indigo-600 dark:text-indigo-400">
                    Scraping Configuration
                </h2>
                
                <form on:submit|preventDefault={handleScrape} class="space-y-6">
                    <div class="space-y-2">
                        <label for="discoverTerms" class="block text-sm font-medium">Discovery Search Terms (comma-separated)</label>
                        <input
                            id="discoverTerms"
                            name="discover_terms" 
                            type="text"
                            bind:value={discoverTerms}
                            class="mt-1 block w-full rounded-lg border-gray-300 shadow-sm p-3 focus:border-indigo-500 focus:ring focus:ring-indigo-500 focus:ring-opacity-50 dark:bg-neutral-800 dark:border-neutral-700"
                            placeholder="e.g., apple, cinnamon, vanilla"
                            required
                            disabled={!!currentJob && currentJob.status === 'RUNNING'}
                        />
                        <p class="text-xs text-gray-500 dark:text-neutral-500">
                            The starting points for the scraper's search.
                        </p>
                    </div>

                    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
                        <label for="maxCompounds" class="block space-y-2">
                            <span class="text-sm font-medium">Max Compounds</span>
                            <input
                                id="maxCompounds"
                                name="max_compounds"
                                type="number"
                                bind:value={maxCompounds}
                                class="block w-full rounded-lg border-gray-300 shadow-sm p-3 focus:border-indigo-500 focus:ring focus:ring-indigo-500 focus:ring-opacity-50 dark:bg-neutral-800 dark:border-neutral-700"
                                min="1"
                                required
                                disabled={!!currentJob && currentJob.status === 'RUNNING'}
                            />
                        </label>
                        
                        <label for="delay" class="block space-y-2">
                            <span class="text-sm font-medium">Delay (in seconds)</span>
                            <input
                                id="delay"
                                name="delay"
                                type="number"
                                bind:value={delay}
                                step="0.01"
                                class="block w-full rounded-lg border-gray-300 shadow-sm p-3 focus:border-indigo-500 focus:ring focus:ring-indigo-500 focus:ring-opacity-50 dark:bg-neutral-800 dark:border-neutral-700"
                                min="0.01"
                                required
                                disabled={!!currentJob && currentJob.status === 'RUNNING'}
                            />
                        </label>
                    </div>

                    {#if !currentJob || currentJob.status !== 'RUNNING'}
                        <div class="text-sm text-gray-500 dark:text-neutral-400 p-2 border-l-4 border-yellow-500 bg-yellow-900/10 rounded">
                            ⚠️ Warning: Job is estimated to take <strong class="text-yellow-400">~{formatTime(estimatedWaitTimeSeconds)}</strong> (or {Math.ceil(estimatedWaitTimeSeconds / 60)} minutes). Higher max compounds and delay increase this time.
                        </div>
                    {/if}
                    
                    <div class="flex gap-4">
                        <button 
                            type="submit" 
                            disabled={isSubmitting || (currentJob && currentJob.status === 'RUNNING')}
                            class="w-full inline-flex border border-transparent transition-colors 
                            items-center justify-center rounded-lg px-8 py-3 font-semibold text-neutral-50 shadow-md 
                            disabled:opacity-50 disabled:cursor-not-allowed mt-4 flex-grow
                            {isSubmitting || (currentJob && currentJob.status === 'RUNNING') ? 'bg-gray-500' : 'bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-600 dark:hover:bg-indigo-500'}"
                        >
                            {#if currentJob && currentJob.status === 'RUNNING'}
                                <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                SCRAPING...
                            {:else if isSubmitting}
                                Starting Job...
                            {:else}
                                Start New Scrape
                            {/if}
                        </button>

                        {#if currentJob && currentJob.status === 'COMPLETED'}
                            <button on:click={handleExport}
                                    type="button" 
                                    class="inline-flex border border-transparent transition-colors items-center justify-center rounded-lg px-8 py-3 font-semibold text-white mt-4 flex-none
                                    bg-green-600 hover:bg-green-700 dark:bg-emerald-600 dark:hover:bg-emerald-500"
                            >
                                Download CSV
                            </button>
                        {/if}
                    </div>
                </form>
            </div>
        </div>
    </main>
</div>

{#if showCompletionModal}
    <div 
        transition:fade={{ duration: 300 }}
        class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-sm"
    >
        <div 
            transition:fly={{ y: -20, duration: 400, easing: quintOut }}
            class="bg-white dark:bg-neutral-800 p-8 rounded-xl shadow-2xl max-w-sm w-full text-center space-y-6"
        >
            <h3 class="text-3xl font-extrabold text-green-600 dark:text-green-400">
                Job Complete! 🎉
            </h3>
            <p class="text-lg text-gray-700 dark:text-neutral-300">
                Scraping job <strong>#{currentJob?.id}</strong> has finished processing. 
                Your dataset is ready for export.
            </p>
            <button 
                on:click={() => { showCompletionModal = false; handleExport(); }}
                class="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-lg transition-colors"
            >
                Download Now
            </button>
            <button 
                on:click={() => showCompletionModal = false}
                class="w-full text-sm text-gray-500 dark:text-neutral-400 mt-2 hover:underline"
            >
                Close (Export Later)
            </button>
        </div>
    </div>
{/if}