document.addEventListener('DOMContentLoaded', function() {
    // 文件名显示
    const fileInput = document.getElementById('file');
    const fileNameDisplay = document.getElementById('file-name');
    
    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                fileNameDisplay.textContent = this.files[0].name;
                fileNameDisplay.style.color = '#333';
            } else {
                fileNameDisplay.textContent = '未选择文件';
                fileNameDisplay.style.color = '#888';
            }
        });
    }
    
    // 表单验证
    const form = document.querySelector('.upload-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            if (!fileInput.files.length) {
                e.preventDefault();
                alert('请选择一个SRT文件上传');
                return false;
            }
            
            // 显示加载中
            const submitBtn = document.querySelector('.submit-btn');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = '翻译中，请稍候...';
            }
            
            return true;
        });
    }
});