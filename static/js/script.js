document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file');
    const fileNameDisplay = document.getElementById('file-name');
    const uploadForm = document.getElementById('upload-form');
    const submitStreamBtn = document.getElementById('submit-stream-btn');
    const submitBackgroundBtn = document.getElementById('submit-background-btn');
    const taskListUl = document.getElementById('task-list');
    const customTermsTextarea = document.getElementById('custom_terms');
    const displayModeRadios = document.querySelectorAll('input[name="display_mode"]');
    const streamCheckbox = document.getElementById('stream'); // Keep for potential future use

    let activePolls = {}; // Store active polling intervals { taskId: intervalId }

    // --- Helper Functions ---
    function getSelectedDisplayMode() {
        for (const radio of displayModeRadios) {
            if (radio.checked) {
                return radio.value;
            }
        }
        return 'only_translated'; // Default
    }

    function addTaskToList(task) {
        if (!taskListUl) return;
        // Avoid adding duplicate tasks if already present
        if (document.getElementById(`task-${task.task_id}`)) {
            console.log(`Task ${task.task_id} already in list, skipping add.`);
            // Optionally update existing entry instead of skipping
            // updateTaskInList(task);
            return;
        }

        const li = document.createElement('li');
        li.id = `task-${task.task_id}`;
        li.dataset.taskId = task.task_id; // Store task_id for easy access
        // Initial state based on provided task data (could be pending, processing, etc.)
        const initialStatus = task.status || 'pending';
        const initialProgress = task.progress || 0;
        li.innerHTML = `
            <div class="task-info">
                <span class="filename">${task.original_filename || '未知文件'}</span>
                <span class="status ${initialStatus}">${getStatusText(initialStatus)}</span>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: ${initialProgress}%; background-color: ${initialStatus === 'failed' ? '#f44336' : (initialStatus === 'completed' ? '#4CAF50' : '#2196F3')};"></div>
                </div>
                <span class="progress-text">${initialProgress}%</span>
                <div class="result-area" style="display: ${initialStatus === 'completed' ? 'block' : 'none'};">${initialStatus === 'completed' ? '翻译完成。点击下载按钮获取结果。' : ''}</div>
                <div class="error-area" style="display: ${initialStatus === 'failed' ? 'block' : 'none'}; color: red;">${initialStatus === 'failed' ? `错误: ${task.error_message || '未知错误'}` : ''}</div>
            </div>
            <div class="task-actions">
                <button class="download-btn" style="display: ${initialStatus === 'completed' ? 'inline-block' : 'none'};">下载结果</button>
                <button class="delete-btn">删除</button>
            </div>
        `;
        // Prepend new tasks to the top for better visibility, or append if loading existing
        if (taskListUl.firstChild && !taskListUl.firstChild.textContent.includes('暂无后台任务')) { // Ensure not inserting before "no tasks" message
            taskListUl.insertBefore(li, taskListUl.firstChild); // Add new tasks to top
        } else {
             if (taskListUl.firstChild && taskListUl.firstChild.textContent.includes('暂无后台任务')) {
                 taskListUl.innerHTML = ''; // Clear "no tasks" message
             }
            taskListUl.appendChild(li); // Add first task or when loading existing
        }

        // --- Add Delete Button Listener ---
        const deleteBtn = li.querySelector('.delete-btn');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', async function() {
                const taskIdToDelete = li.dataset.taskId;
                if (confirm(`确定要删除任务 "${task.original_filename || taskIdToDelete}" 吗？`)) {
                    console.log(`Attempting to delete task ${taskIdToDelete}`);
                    try {
                        const response = await fetch(`/task/${taskIdToDelete}`, {
                            method: 'DELETE'
                        });

                        if (response.ok || response.status === 204) { // 204 No Content is success for DELETE
                            console.log(`Task ${taskIdToDelete} deleted successfully.`);
                            stopPolling(taskIdToDelete); // Stop polling if it was active
                            taskListUl.removeChild(li); // Remove from UI
                            // Check if list is now empty
                            if (taskListUl.children.length === 0) {
                                taskListUl.innerHTML = '<li>暂无后台任务。</li>';
                            }
                        } else {
                            let errorMsg = `删除失败 (${response.status}): ${response.statusText}`;
                            try {
                                const errData = await response.json();
                                errorMsg = `删除失败: ${errData.detail || response.statusText}`;
                            } catch (e) { /* Ignore if error response is not JSON */ }
                            console.error(`Failed to delete task ${taskIdToDelete}: ${errorMsg}`);
                            alert(`删除任务失败: ${errorMsg}`);
                        }
                    } catch (error) {
                        console.error(`Network error deleting task ${taskIdToDelete}:`, error);
                        alert(`删除任务时发生网络错误: ${error.message}`);
                    }
                }
            });
        }
        // --- End Delete Button Listener ---


        // Update download button functionality if completed
        if (initialStatus === 'completed') {
            const downloadBtn = li.querySelector('.download-btn');
            // Fetch full task data to get the result for download
            fetch(`/task/${task.task_id}`)
                .then(response => response.ok ? response.json() : Promise.reject('Failed to fetch result'))
                .then(fullTaskData => {
                    if (fullTaskData.result !== undefined && fullTaskData.result !== null) {
                        downloadBtn.onclick = () => downloadResult(fullTaskData.result, task.original_filename);
                    } else {
                        li.querySelector('.result-area').textContent = "翻译完成，但结果数据丢失。";
                        downloadBtn.style.display = 'none';
                    }
                })
                .catch(error => {
                    console.error(`Error fetching result for completed task ${task.task_id}:`, error);
                    li.querySelector('.result-area').textContent = "翻译完成，但获取结果失败。";
                    downloadBtn.style.display = 'none';
                });
        }

        // Start polling only if the task is not in a final state
        if (initialStatus !== 'completed' && initialStatus !== 'failed') {
            // Check if polling is already active before starting a new one
            if (!activePolls[task.task_id]) {
                 activePolls[task.task_id] = setTimeout(() => pollTaskStatus(task.task_id), 500); // Start polling after a short delay
            }
        }
    }


    function updateTaskInList(taskData) {
        const taskLi = document.getElementById(`task-${taskData.task_id}`);
        if (!taskLi) return;

        const statusSpan = taskLi.querySelector('.status');
        const progressBar = taskLi.querySelector('.progress-bar');
        const progressText = taskLi.querySelector('.progress-text');
        const resultArea = taskLi.querySelector('.result-area');
        const errorArea = taskLi.querySelector('.error-area');
        const downloadBtn = taskLi.querySelector('.download-btn');
        const deleteBtn = taskLi.querySelector('.delete-btn'); // Get delete button reference

        // Update Status
        statusSpan.textContent = getStatusText(taskData.status);
        statusSpan.className = `status ${taskData.status}`; // Update class for styling

        // Update Progress
        const progress = taskData.progress || 0;
        progressBar.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;

        // Handle final states
        if (taskData.status === 'completed') {
            progressBar.style.backgroundColor = '#4CAF50'; // Green for completed
            resultArea.textContent = "翻译完成。点击下载按钮获取结果。"; // Inform user
            resultArea.style.display = 'block';
            errorArea.style.display = 'none';
            downloadBtn.style.display = 'inline-block';
            deleteBtn.style.display = 'inline-block'; // Ensure delete button is visible
            // Ensure result is available before setting onclick
            if (taskData.result !== undefined && taskData.result !== null) {
                 downloadBtn.onclick = () => downloadResult(taskData.result, taskData.original_filename);
            } else {
                 // Handle case where result might be missing unexpectedly (e.g., if fetched from /tasks list initially)
                 // Fetch full data again to be sure
                 fetch(`/task/${taskData.task_id}`)
                    .then(response => response.ok ? response.json() : Promise.reject('Failed to fetch result'))
                    .then(fullTaskData => {
                        if (fullTaskData.result !== undefined && fullTaskData.result !== null) {
                            downloadBtn.onclick = () => downloadResult(fullTaskData.result, taskData.original_filename);
                        } else {
                            resultArea.textContent = "翻译完成，但结果数据丢失。";
                            downloadBtn.style.display = 'none';
                        }
                    })
                    .catch(error => {
                        console.error(`Error fetching result for completed task ${taskData.task_id}:`, error);
                        resultArea.textContent = "翻译完成，但获取结果失败。";
                        downloadBtn.style.display = 'none';
                    });
            }
            stopPolling(taskData.task_id); // Stop polling
        } else if (taskData.status === 'failed') {
            progressBar.style.backgroundColor = '#f44336'; // Red for failed
            errorArea.textContent = `错误: ${taskData.error_message || '未知错误'}`;
            errorArea.style.display = 'block';
            resultArea.style.display = 'none';
            downloadBtn.style.display = 'none';
            deleteBtn.style.display = 'inline-block'; // Ensure delete button is visible
            stopPolling(taskData.task_id); // Stop polling
        } else {
             // Keep polling for pending/processing
             progressBar.style.backgroundColor = '#2196F3'; // Blue for processing/pending
             resultArea.style.display = 'none';
             errorArea.style.display = 'none';
             downloadBtn.style.display = 'none';
             deleteBtn.style.display = 'inline-block'; // Ensure delete button is visible
             scheduleNextPoll(taskData.task_id);
        }
    }

    function getStatusText(status) {
        switch (status) {
            case 'pending': return '等待中...';
            case 'processing': return '处理中...';
            case 'completed': return '已完成';
            case 'failed': return '失败';
            default: return status;
        }
    }

    function downloadResult(text, originalFilename) {
        // Ensure text is a string
        const content = (typeof text === 'string') ? text : JSON.stringify(text);

        // New filename logic: basename-ch.srt
        let baseName = originalFilename || 'task_result';
        const lastDotIndex = baseName.lastIndexOf('.');
        if (lastDotIndex > 0) { // Check if dot exists and is not the first character
            baseName = baseName.substring(0, lastDotIndex);
        }
        let safeFilename = `${baseName}-ch.srt`;

        // Basic sanitization (remove potentially problematic characters)
        safeFilename = safeFilename.replace(/[/\\]/g, '_'); // Replace slashes with underscore
        safeFilename = safeFilename.replace(/[:*?"<>|]/g, '_'); // Replace other common invalid chars

        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = safeFilename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href); // Clean up
    }

    // --- Polling Logic ---
    async function pollTaskStatus(taskId) {
        console.log(`Polling status for task ${taskId}`);
        // Ensure polling wasn't stopped while waiting for the fetch
        if (!activePolls[taskId]) {
             console.log(`Polling already stopped for task ${taskId} before fetch completed.`);
             return;
        }
        try {
            const response = await fetch(`/task/${taskId}`);
            // Check again if polling was stopped after fetch completed
            if (!activePolls[taskId]) {
                 console.log(`Polling stopped for task ${taskId} after fetch completed.`);
                 return;
            }

            if (!response.ok) {
                console.error(`Error fetching status for task ${taskId}: ${response.status} ${response.statusText}`);
                if (response.status === 404) {
                     // If task not found during polling, assume it was deleted elsewhere
                     console.warn(`Task ${taskId} not found during polling, likely deleted.`);
                     stopPolling(taskId); // Stop polling
                     const taskLi = document.getElementById(`task-${taskId}`);
                     if (taskLi) {
                         taskListUl.removeChild(taskLi); // Remove from UI
                         if (taskListUl.children.length === 0) {
                             taskListUl.innerHTML = '<li>暂无后台任务。</li>';
                         }
                     }
                } else {
                     // Schedule retry for other server errors
                     scheduleNextPoll(taskId, 5000);
                }
                return; // Stop processing this poll cycle
            }

            const taskData = await response.json();
            // Check one last time before updating UI
            if (!activePolls[taskId]) {
                 console.log(`Polling stopped for task ${taskId} before updating UI.`);
                 return;
            }
            updateTaskInList(taskData); // This function now handles scheduling the next poll or stopping

        } catch (error) {
            console.error(`Network error polling task ${taskId}:`, error);
             // Check if polling was stopped during the error handling
            if (!activePolls[taskId]) {
                 console.log(`Polling stopped for task ${taskId} during error handling.`);
                 return;
            }
            scheduleNextPoll(taskId, 5000); // Retry after longer delay on network error
        }
    }

    function scheduleNextPoll(taskId, delay = 2000) {
         // Ensure polling is still intended for this task
         if (activePolls[taskId]) {
             // Clear previous timeout just in case this function is called rapidly
             clearTimeout(activePolls[taskId]);
             activePolls[taskId] = setTimeout(() => pollTaskStatus(taskId), delay);
             console.log(`Scheduled next poll for task ${taskId} in ${delay}ms`);
         } else {
             console.log(`Not scheduling next poll for task ${taskId} as polling is stopped.`);
         }
    }


    function stopPolling(taskId) {
        if (activePolls[taskId]) {
            clearTimeout(activePolls[taskId]);
            delete activePolls[taskId];
            console.log(`Stopped polling for task ${taskId}`);
        }
    }

    // --- Function to Load Existing Tasks on Page Load ---
    async function loadExistingTasks() {
        console.log("Loading existing tasks...");
        try {
            const response = await fetch('/tasks'); // Fetch from the new endpoint
            if (!response.ok) {
                console.error(`Error fetching existing tasks: ${response.status} ${response.statusText}`);
                // Optionally display an error message to the user
                taskListUl.innerHTML = '<li>加载任务列表失败。</li>';
                return;
            }
            const result = await response.json();
            if (result.tasks && Array.isArray(result.tasks)) {
                console.log(`Found ${result.tasks.length} existing tasks.`);
                taskListUl.innerHTML = ''; // Clear any placeholder content
                if (result.tasks.length === 0) {
                     taskListUl.innerHTML = '<li>暂无后台任务。</li>';
                } else {
                    result.tasks.forEach(taskData => {
                        addTaskToList(taskData); // Add each task to the list
                    });
                }
            } else {
                 console.warn("No tasks found or invalid response format from /tasks");
                 taskListUl.innerHTML = '<li>暂无后台任务。</li>';
            }
        } catch (error) {
            console.error('Network error loading existing tasks:', error);
            taskListUl.innerHTML = '<li>加载任务列表时发生网络错误。</li>';
        }
    }


    // --- Event Listeners ---

    // Update file name display for single or multiple files
    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 1) {
                fileNameDisplay.textContent = `${this.files.length} 个文件已选择`;
                fileNameDisplay.style.color = '#333';
            } else if (this.files.length === 1) {
                fileNameDisplay.textContent = this.files[0].name;
                fileNameDisplay.style.color = '#333';
            } else {
                fileNameDisplay.textContent = '未选择文件';
                fileNameDisplay.style.color = '#888';
            }
        });
    }

    // Stream Task Button Listener (Modified Logic)
    if (submitStreamBtn && uploadForm) {
        console.log("Stream button listener attached.");
        submitStreamBtn.addEventListener('click', function() {
            console.log("Stream button clicked.");

            if (!fileInput.files || fileInput.files.length !== 1) {
                console.log("Stream button validation failed: Incorrect file count.");
                alert('请选择一个SRT文件进行实时翻译。');
                return;
            }
            const file = fileInput.files[0];
            console.log("Stream button validation passed.");

            const displayMode = getSelectedDisplayMode();
            const customTerms = customTermsTextarea.value;
            const filename = file.name; // Get filename

            console.log(`Storing data for stream: filename=${filename}, displayMode=${displayMode}, customTerms=${customTerms}`);

            // Store necessary data in sessionStorage
            try {
                sessionStorage.setItem('streamFilename', filename);
                sessionStorage.setItem('streamDisplayMode', displayMode);
                sessionStorage.setItem('streamCustomTerms', customTerms);
                console.log("Data stored in sessionStorage.");

                // Navigate to the stream page
                console.log("Navigating to /stream.html");
                window.location.href = '/stream.html'; // Navigate to the static HTML page

            } catch (e) {
                console.error("Error storing data in sessionStorage or navigating:", e);
                alert("启动流式翻译时出错，可能是浏览器限制或存储已满。");
                // Optionally re-enable button here if needed
                submitStreamBtn.disabled = false;
                submitStreamBtn.textContent = '实时翻译 (单文件)';
            }
        });
    } else {
        console.error("Could not find Stream button or Upload form elements.");
    }


    // Background Task Button Listener
    if (submitBackgroundBtn) {
        submitBackgroundBtn.addEventListener('click', async function() {
            console.log("Background button clicked."); // Log BG 1
            if (!fileInput.files || fileInput.files.length === 0) {
                console.log("Background validation failed: No files selected."); // Log BG 2a
                alert('请至少选择一个SRT文件上传');
                return;
            }
            console.log(`Background validation passed: ${fileInput.files.length} file(s) selected.`); // Log BG 2b

            submitBackgroundBtn.disabled = true;
            submitBackgroundBtn.textContent = '提交中...';
            console.log("Background button disabled."); // Log BG 3

            const formData = new FormData();
            const displayMode = getSelectedDisplayMode();
            const customTerms = customTermsTextarea.value;

            formData.append('display_mode', displayMode);
            formData.append('custom_terms', customTerms);
            console.log("Appended display_mode and custom_terms to FormData."); // Log BG 4

            let url = '';
            // Use correct name ('file' vs 'files') based on count
            if (fileInput.files.length === 1) {
                url = '/upload';
                // Ensure file input name is 'file' for single upload endpoint
                if (fileInput.name !== 'file') {
                    console.warn(`File input name is '${fileInput.name}', temporarily using 'file' for single background upload.`);
                    // We add the file with the correct key directly to FormData
                    formData.append('file', fileInput.files[0]);
                } else {
                    formData.append(fileInput.name, fileInput.files[0]);
                }
                console.log(`Prepared for single file upload to ${url}`); // Log BG 5a
            } else {
                url = '/upload-multiple';
                 // Ensure file input name is 'files' for multiple upload endpoint
                if (fileInput.name !== 'files') {
                     console.warn(`File input name is '${fileInput.name}', temporarily using 'files' for multiple background upload.`);
                     // Add files with the correct key directly to FormData
                     for (let i = 0; i < fileInput.files.length; i++) {
                         formData.append('files', fileInput.files[i]);
                     }
                } else {
                     for (let i = 0; i < fileInput.files.length; i++) {
                         formData.append(fileInput.name, fileInput.files[i]);
                     }
                }
                console.log(`Prepared for multiple file upload to ${url}`); // Log BG 5b
            }

            try {
                console.log(`Sending fetch request to ${url}`); // Log BG 6
                const response = await fetch(url, {
                    method: 'POST',
                    body: formData
                });
                console.log(`Received response from ${url}, status: ${response.status}`); // Log BG 7

                if (!response.ok) {
                    let errorMsg = `提交失败 (${response.status}): ${response.statusText}`;
                    let errorDetail = '';
                    try {
                        const errData = await response.json();
                        errorDetail = errData.detail || '';
                        errorMsg = `提交失败: ${errorDetail || response.statusText}`;
                    } catch (e) {
                        console.warn("Could not parse error response as JSON.");
                    }
                    console.error(`Fetch error: ${errorMsg}`, errorDetail); // Log BG 8a
                    throw new Error(errorMsg);
                }

                const result = await response.json();
                console.log('Tasks submitted response:', result); // Log BG 8b

                if (result.tasks && Array.isArray(result.tasks) && result.tasks.length > 0) {
                     console.log(`Processing ${result.tasks.length} tasks from response.`); // Log BG 9a
                     // Clear the "no tasks" message if it exists
                     const noTasksLi = taskListUl.querySelector('li');
                     if (noTasksLi && noTasksLi.textContent.includes('暂无后台任务')) {
                         taskListUl.innerHTML = '';
                     }
                     result.tasks.forEach(taskInfo => {
                         if (taskInfo.task_id && taskInfo.filename) {
                             const taskDataForUI = {
                                 task_id: taskInfo.task_id,
                                 original_filename: taskInfo.filename,
                                 status: 'pending',
                                 progress: 0
                             };
                             addTaskToList(taskDataForUI);
                         } else {
                             console.warn("Received incomplete task info:", taskInfo);
                         }
                     });
                     // Clear file input after successful submission
                     fileInput.value = '';
                     fileNameDisplay.textContent = '未选择文件';
                     fileNameDisplay.style.color = '#888';
                     console.log("Cleared file input."); // Log BG 9b
                } else {
                     console.error("Invalid task submission response:", result); // Log BG 9c
                     alert('任务提交似乎成功，但服务器未返回有效的任务信息。请检查服务器日志。');
                }

            } catch (error) {
                console.error('Background task submission error:', error); // Log BG 10
                alert(`任务提交出错: ${error.message}`);
            } finally {
                submitBackgroundBtn.disabled = false;
                submitBackgroundBtn.textContent = '提交后台任务';
                console.log("Background button re-enabled."); // Log BG 11
                // Restore original file input name if it was changed (Not strictly necessary as we use FormData keys)
                // if (fileInput.files.length === 1 && fileInput.name !== 'file') {
                //     fileInput.name = 'files'; // Assuming original was 'files' based on HTML
                // } else if (fileInput.files.length > 1 && fileInput.name !== 'files') {
                //      fileInput.name = 'files'; // Assuming original was 'files'
                // }
            }
        });
    } else {
         console.error("Could not find Background button element.");
    }

    // --- Initial Load ---
    loadExistingTasks(); // Load tasks when the page loads

});