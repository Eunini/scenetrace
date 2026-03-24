#!/bin/sh
case " ${RENEWED_DOMAINS:-} " in
  *" scenetrace.193-203-15-98.sslip.io "*)
    /usr/sbin/nginx -t || exit 1
    /usr/bin/systemctl reload nginx
    ;;
esac
