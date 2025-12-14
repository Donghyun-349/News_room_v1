"""
이메일 발송 모듈
생성된 보고서를 이메일로 발송하는 모듈
"""
import os
import sys
import io
import logging
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import smtplib
from datetime import datetime
from typing import List, Dict, Any, Optional

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logger = logging.getLogger(__name__)


class EmailSender:
    """이메일 발송 클래스"""
    
    def __init__(self):
        """이메일 발송자 초기화"""
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.email_from = os.getenv('EMAIL_FROM', self.smtp_user)
        
        if not all([self.smtp_user, self.smtp_password]):
            logger.warning("이메일 설정이 완료되지 않았습니다. SMTP_USER, SMTP_PASSWORD를 확인하세요.")
    
    def markdown_to_html(self, markdown_text: str) -> str:
        """Markdown 텍스트를 간단한 HTML로 변환"""
        import re
        html = markdown_text
        
        # 코드 블록 보호 (```...```)
        code_blocks = []
        def code_block_replacer(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        html = re.sub(r'```[\s\S]*?```', code_block_replacer, html)
        
        # 인라인 코드 보호 (`...`)
        inline_codes = []
        def inline_code_replacer(match):
            inline_codes.append(match.group(0))
            return f"__INLINE_CODE_{len(inline_codes)-1}__"
        html = re.sub(r'`([^`]+)`', inline_code_replacer, html)
        
        # 헤더 변환 (줄 시작에서만)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        
        # 강조 변환 **텍스트**
        html = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', html)
        
        # 링크 변환 [텍스트](URL)
        html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', html)
        
        # 리스트 변환 (- 로 시작하는 줄)
        lines = html.split('\n')
        in_list = False
        result_lines = []
        for line in lines:
            if re.match(r'^- (.+)$', line):
                if not in_list:
                    result_lines.append('<ul>')
                    in_list = True
                result_lines.append(f'<li>{re.sub(r"^- (.+)$", r"\\1", line)}</li>')
            else:
                if in_list:
                    result_lines.append('</ul>')
                    in_list = False
                result_lines.append(line)
        if in_list:
            result_lines.append('</ul>')
        html = '\n'.join(result_lines)
        
        # 구분선 변환 (---)
        html = re.sub(r'^---$', r'<hr>', html, flags=re.MULTILINE)
        
        # 코드 블록 복원
        for i, code_block in enumerate(code_blocks):
            code_content = code_block.replace('```', '').strip()
            html = html.replace(f"__CODE_BLOCK_{i}__", f'<pre><code>{code_content}</code></pre>')
        
        # 인라인 코드 복원
        for i, inline_code in enumerate(inline_codes):
            code_content = inline_code.replace('`', '')
            html = html.replace(f"__INLINE_CODE_{i}__", f'<code>{code_content}</code>')
        
        # 줄바꿈 변환 (연속된 줄바꿈은 <p> 태그로)
        html = re.sub(r'\n\n+', '</p><p>', html)
        html = '<p>' + html + '</p>'
        html = html.replace('\n', '<br>\n')
        
        return html
    
    def send_report(self, recipient: Dict[str, Any], report_name: str, report_content: str, report_path: Optional[Path] = None) -> bool:
        """
        보고서를 이메일로 발송합니다.
        
        Args:
            recipient: 수신자 정보 {'email': str, 'name': str, 'report_groups': List[str]}
            report_name: 보고서 이름
            report_content: 보고서 내용 (Markdown 형식)
            report_path: 보고서 파일 경로 (선택사항, 첨부용)
        
        Returns:
            발송 성공 여부
        """
        if not all([self.smtp_user, self.smtp_password]):
            logger.error("이메일 설정이 완료되지 않았습니다.")
            return False
        
        recipient_email = recipient.get('email', '')
        recipient_name = recipient.get('name', recipient_email.split('@')[0])
        
        if not recipient_email:
            logger.error("수신자 이메일이 없습니다.")
            return False
        
        try:
            # 이메일 생성
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_from
            msg['To'] = recipient_email
            msg['Subject'] = Header(f"Daily Market Report - {report_name} - {datetime.now().strftime('%Y-%m-%d')}", 'utf-8')
            
            # 본문 작성 (HTML 형식)
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        ul {{ margin: 10px 0; padding-left: 20px; }}
        li {{ margin: 5px 0; }}
        strong {{ color: #2c3e50; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
{self.markdown_to_html(report_content)}
</body>
</html>
"""
            
            # HTML 본문 추가
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # 파일 첨부 (선택사항)
            if report_path and report_path.exists():
                with open(report_path, 'rb') as f:
                    from email.mime.base import MIMEBase
                    from email import encoders
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {report_path.name}'
                    )
                    msg.attach(part)
            
            # 이메일 발송
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            text = msg.as_string()
            server.sendmail(self.email_from, recipient_email, text)
            server.quit()
            
            logger.info(f"✅ 이메일 발송 완료: {recipient_name} ({recipient_email}) - {report_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 이메일 발송 실패: {recipient_email} - {e}", exc_info=True)
            return False
    
    def send_reports_to_recipients(self, report_files: Dict[str, Path]) -> Dict[str, Any]:
        """
        생성된 보고서를 수신자별로 발송합니다.
        
        Args:
            report_files: {report_group_name: report_file_path} 딕셔너리
        
        Returns:
            발송 결과 통계
        """
        from modules.email_recipient_loader import EmailRecipientLoader
        
        recipient_loader = EmailRecipientLoader()
        recipients = recipient_loader.get_all()
        
        if not recipients:
            logger.warning("이메일 수신자가 없습니다.")
            return {
                'total_recipients': 0,
                'sent_count': 0,
                'failed_count': 0,
                'details': []
            }
        
        results = {
            'total_recipients': len(recipients),
            'sent_count': 0,
            'failed_count': 0,
            'details': []
        }
        
        logger.info(f"이메일 발송 시작: {len(recipients)}명 수신자, {len(report_files)}개 보고서")
        
        for recipient in recipients:
            recipient_email = recipient.get('email', '')
            recipient_name = recipient.get('name', recipient_email.split('@')[0])
            report_groups = recipient.get('report_groups', [])
            
            if not report_groups:
                logger.info(f"  ⏭️  {recipient_name}: 보고서 그룹이 지정되지 않아 건너뜁니다.")
                continue
            
            # 수신자가 받을 보고서 찾기
            sent_reports = []
            for report_group_name, report_path in report_files.items():
                if report_group_name in report_groups:
                    # 보고서 내용 읽기
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            report_content = f.read()
                        
                        # 이메일 발송
                        success = self.send_report(
                            recipient=recipient,
                            report_name=report_group_name,
                            report_content=report_content,
                            report_path=report_path
                        )
                        
                        if success:
                            sent_reports.append(report_group_name)
                            results['sent_count'] += 1
                        else:
                            results['failed_count'] += 1
                    except Exception as e:
                        logger.error(f"보고서 읽기 실패: {report_path} - {e}")
                        results['failed_count'] += 1
            
            if sent_reports:
                results['details'].append({
                    'recipient': recipient_name,
                    'email': recipient_email,
                    'reports': sent_reports,
                    'status': 'success'
                })
                logger.info(f"  ✅ {recipient_name}: {', '.join(sent_reports)} 발송 완료")
            else:
                results['details'].append({
                    'recipient': recipient_name,
                    'email': recipient_email,
                    'reports': [],
                    'status': 'no_reports'
                })
        
        logger.info(f"이메일 발송 완료: 총 {results['sent_count']}개 발송, {results['failed_count']}개 실패")
        return results

